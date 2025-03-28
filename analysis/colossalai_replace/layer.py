import dataclasses
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from colossalai.moe._operation import AllGather, AllToAll, HierarchicalAllToAll, MoeCombine, MoeDispatch, ReduceScatter
from colossalai.moe.experts import MLPExperts
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.routers import MoeRouter, get_router_cls
from colossalai.moe.utils import create_ep_hierarchical_group, get_noise_generator
from colossalai.tensor.moe_tensor.api import get_dp_group, get_ep_group, get_ep_group_ranks, get_ep_size

import json
import numpy as np


class SparseMLPWithLayerSkip(nn.Module):
    """A class for users to create MoE modules with LayerSkip functionality.

    This enhances the original SparseMLP with early exit capabilities, allowing the model
    to dynamically skip processing in subsequent layers for tokens that already have
    high-confidence representations.

    Args:
        dim_model (int): Hidden dimension of training model
        num_experts (int): The number experts
        top_k (int, optional): The number of experts for dispatchment of each token
        capacity_factor_train (float, optional): Capacity factor in routing during training
        capacity_factor_eval (float, optional): Capacity factor in routing during evaluation
        min_capacity (int, optional): The minimum number of the capacity of each expert
        noisy_policy (str, optional): The policy of noisy function. Now we have 'Jitter' and 'Gaussian'.
        drop_tks (bool, optional): Whether drops tokens in evaluation
        enable_layerskip (bool, optional): Whether to enable the LayerSkip functionality
        confidence_threshold (float, optional): Threshold for early exit confidence score
        expert_dropout_rate (float, optional): Dropout rate for experts during training
        auxiliary_loss_weight (float, optional): Weight for auxiliary loss components
    """

    def __init__(
            self,
            num_experts: int,
            hidden_size: int,
            intermediate_size: int,
            router_top_k: int = 1,
            router_capacity_factor_train: float = 1.25,
            router_capacity_factor_eval: float = 2.0,
            router_min_capacity: int = 4,
            router_noisy_policy: Optional[str] = None,
            router_drop_tks: bool = True,
            mlp_activation: Optional[str] = None,
            mlp_gated: bool = False,
            enable_load_balance: bool = False,
            load_balance_tolerance: float = 0.1,
            load_balance_beam_width: int = 8,
            load_balance_group_swap_factor: float = 0.4,
            enable_kernel: bool = False,
            enable_comm_overlap: bool = False,
            enable_hierarchical_comm: bool = False,
            model_output_dir: str = None,
            # LayerSkip specific parameters
            enable_layerskip: bool = True,
            confidence_threshold: float = 0.8,
            expert_dropout_rate: float = 0.2,
            auxiliary_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.gated = mlp_gated
        self.enable_kernel = enable_kernel
        self.enable_comm_overlap = enable_comm_overlap
        self.expert_parallel = MOE_MANAGER.get_parallel()
        self.model_output_dir = model_output_dir

        # LayerSkip specific variables
        self.enable_layerskip = enable_layerskip
        self.confidence_threshold = confidence_threshold
        self.expert_dropout_rate = expert_dropout_rate
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.training_mode = True  # Tracks if we're in training or evaluation mode

        # For MoE Analysis
        if self.model_output_dir is not None:
            self.output_json_file = open(f"{self.model_output_dir}/output.json", "w")

        # moe router
        noisy_func = get_noise_generator(router_noisy_policy, num_experts)
        router_cls = get_router_cls(router_top_k)
        self.topk = router_top_k
        self.router: MoeRouter = router_cls(
            capacity_factor_train=router_capacity_factor_train,
            capacity_factor_eval=router_capacity_factor_eval,
            min_capacity=router_min_capacity,
            noisy_func=noisy_func,
            drop_tks=router_drop_tks,
        )

        # gate
        self.gate_weight = torch.nn.Parameter(torch.empty(num_experts, self.hidden_size))

        # moe experts
        self.experts = MLPExperts(
            num_experts=self.num_experts,
            expert_parallel=self.expert_parallel,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=mlp_activation,
            gated=mlp_gated,
            use_kernel=self.enable_kernel,
        )

        # get parallel settings
        if self.expert_parallel is not None:
            self.ep_group = get_ep_group(self.experts)
            self.ep_size = get_ep_size(self.experts)
            self.ep_hierarchical_group = None
            if enable_hierarchical_comm:
                self.ep_intra_src_rank, *self.ep_hierarchical_group = create_ep_hierarchical_group(
                    get_ep_group_ranks(self.experts)
                )
            self.dp_group = get_dp_group(self.experts)
        else:
            self.ep_group = None
            self.dp_group = None
        self.num_local_experts = self.experts.num_local_experts

        # load balance
        self.enable_load_balance = enable_load_balance
        if self.enable_load_balance == True:
            from colossalai.moe.load_balance import LoadBalancer
            self.load_balancer = LoadBalancer(
                experts=self.experts,
                gate=self.gate_weight,
                local_expert_num=self.num_local_experts,
                expert_num=self.num_experts,
                ep_group=self.ep_group,
                dp_group=self.dp_group,
                tolerance=load_balance_tolerance,
                beam_width=load_balance_beam_width,
                group_swap_factor=load_balance_group_swap_factor,
            )

        # LayerSkip components
        if self.enable_layerskip:
            # Confidence predictor network
            self.confidence_predictor = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 2, 1)
            )

            # Input complexity estimator - helps determine if input is "easy" or "hard"
            self.complexity_estimator = nn.Linear(self.hidden_size, 1)

            # Register buffer to track expert utilization for load balancing
            self.register_buffer('expert_utilization', torch.zeros(num_experts))

        # Layer normalization for input (helps with early exit decisions)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # init param
        self.reset_parameters()

        # Early exit statistics tracking
        self.early_exit_stats = {
            'total_tokens': 0,
            'early_exits': 0,
            'confidence_scores': [],
            'complexity_scores': []
        }

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.normal_(self.gate_weight, std=math.sqrt(0.1 / self.hidden_size))
        if self.enable_layerskip:
            # Initialize confidence predictor with slightly conservative values
            # (biased toward not exiting early)
            for module in self.confidence_predictor.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, -1.0)  # Initially conservative

    def reset_utilization_stats(self):
        """Reset the expert utilization tracking."""
        if self.enable_layerskip:
            self.expert_utilization.zero_()
            self.early_exit_stats = {
                'total_tokens': 0,
                'early_exits': 0,
                'confidence_scores': [],
                'complexity_scores': []
            }

    def forward(self, inputs: torch.Tensor, return_metrics: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Args:
            inputs (torch.Tensor): The input tensor of shape (batch_size, seq_len, hidden_size)
            return_metrics (bool): Whether to return additional metrics

        Returns:
            torch.Tensor or Tuple[torch.Tensor, dict]: The output tensor of shape (batch_size, seq_len, hidden_size)
                and optionally metrics for analysis
        """
        # reshape the input tokens
        tokens = inputs.reshape(-1, self.hidden_size)
        batch_size = tokens.shape[0]

        # Store original input for residual connection and early exit
        residual = tokens

        # Apply layer normalization (helps with confidence prediction)
        tokens = self.layer_norm(tokens)

        # Calculate input complexity if LayerSkip is enabled
        complexity_score = None
        if self.enable_layerskip:
            complexity_score = torch.sigmoid(self.complexity_estimator(tokens)).mean()

        # the data type of the inputs in the gating should be fp32
        fp32_input = tokens.to(torch.float)
        fp32_weight = self.gate_weight.to(torch.float)
        gate_output = F.linear(fp32_input, fp32_weight)

        # Apply expert dropout during training with dynamic rate based on complexity
        if self.training and self.expert_dropout_rate > 0 and self.enable_layerskip:
            # Create a mask for expert dropout
            expert_mask = torch.rand(self.num_experts, device=tokens.device) > self.expert_dropout_rate

            # Ensure at least top_k experts are active
            if expert_mask.sum() < self.topk:
                inactive_indices = torch.where(~expert_mask)[0]
                to_activate = min(self.topk - expert_mask.sum().item(), len(inactive_indices))
                if to_activate > 0:
                    indices_to_activate = inactive_indices[torch.randperm(len(inactive_indices))[:to_activate]]
                    expert_mask[indices_to_activate] = True

            # Apply mask to gate output by setting logits for dropped experts to a large negative value
            masked_gate_output = gate_output.clone()
            masked_gate_output[:, ~expert_mask] = -1e10
            gate_output = masked_gate_output

        # update expert load for load balancing
        if self.enable_load_balance == True:
            with torch.no_grad():
                # TODO: optimize computation
                expert_load = torch.topk(gate_output, k=self.topk, dim=-1)[1]
                # TODO: bincount introduces synchronize, fix it
                expert_load = torch.bincount(expert_load.view(-1))
                self.load_balancer.update_load(expert_load)

        # the result from the router
        used_capacity, *route_result_list = self.router(
            inputs=gate_output, use_kernel=self.enable_kernel, ep_group=self.ep_group)

        # For LayerSkip - track which experts are used and update utilization
        if self.enable_layerskip and self.training:
            # Extract top-k expert indices
            top_indices = route_result_list[1]  # shape: [topk, batch_size, capacity]
            # Flatten to get all expert indices
            expert_indices = top_indices.reshape(-1)
            # Update expert utilization
            for idx in expert_indices:
                self.expert_utilization[idx] += 1

        # Check for early exit condition
        early_exit = False
        confidence_score = None
        early_exit_metadata = {}

        if self.enable_layerskip and not self.training:
            # Predict confidence for early exit
            confidence_logits = self.confidence_predictor(tokens)
            confidence_score = torch.sigmoid(confidence_logits).mean()

            # Adjust confidence threshold based on complexity
            if complexity_score is not None:
                # Make threshold higher for complex inputs (harder to exit early)
                adjusted_threshold = self.confidence_threshold * (1.0 + 0.2 * complexity_score)
            else:
                adjusted_threshold = self.confidence_threshold

            # Check if confidence exceeds threshold
            if confidence_score > adjusted_threshold:
                early_exit = True

                # Store statistics
                self.early_exit_stats['total_tokens'] += batch_size
                self.early_exit_stats['early_exits'] += batch_size
                self.early_exit_stats['confidence_scores'].append(confidence_score.item())
                if complexity_score is not None:
                    self.early_exit_stats['complexity_scores'].append(complexity_score.item())

                # Prepare early exit metadata
                early_exit_metadata = {
                    'early_exit': True,
                    'confidence_score': confidence_score.item(),
                    'complexity_score': complexity_score.item() if complexity_score is not None else None,
                    'threshold': adjusted_threshold
                }

                # Skip expert computation and return
                if return_metrics:
                    return residual, early_exit_metadata
                return residual

        # If no early exit, update stats
        if self.enable_layerskip and not self.training:
            self.early_exit_stats['total_tokens'] += batch_size

        # Convert variables to NumPy arrays for analysis if output directory is specified
        if self.model_output_dir is not None:
            gate_output_np = gate_output.detach().cpu().numpy()
            used_capacity_np = used_capacity.detach().cpu().numpy()
            dispatch_mask_np = route_result_list[1].detach().cpu().numpy()
            combine_score_np = route_result_list[0].detach().cpu().numpy()

            # Create a dictionary to store the NumPy arrays
            data = {
                "gate_output": gate_output_np.tolist(),
                "used_capacity": used_capacity_np.tolist(),
                "dispatch_mask": dispatch_mask_np.tolist(),
                "combine_score": combine_score_np.tolist()
            }

            # Add LayerSkip metrics if available
            if self.enable_layerskip:
                if confidence_score is not None:
                    data["confidence_score"] = confidence_score.item()
                if complexity_score is not None:
                    data["complexity_score"] = complexity_score.item()
                data["early_exit"] = early_exit

            # Save the dictionary to the output JSON file
            json.dump(data, self.output_json_file)
            self.output_json_file.write('\n')

        # dispatch_data: (num_experts, capacity, hidden_size)
        if self.enable_kernel:
            dispatch_data = MoeDispatch.apply(tokens, *route_result_list[1:])
            dispatch_data = dispatch_data.reshape(self.num_experts, -1, self.hidden_size)
        else:
            sec_mask_f = route_result_list[1].type_as(inputs)
            dispatch_data = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)

        # expert_output: (num_groups, num_experts, capacity, hidden_size)
        if self.expert_parallel == "EP":
            expert_output = self._ep_process(
                dispatch_data,
                used_capacity,
                overlap=self.enable_comm_overlap
            )
        elif self.expert_parallel == "TP":
            expert_output = self._tp_process(
                dispatch_data,
                used_capacity,
                overlap=self.enable_comm_overlap
            )
        elif self.expert_parallel is None:
            expert_output = self._local_process(dispatch_data)
        else:
            raise NotImplementedError("This kind of communication has not been implemented yet.\n"
                                      "Please use Experts build function.")

        if self.enable_kernel:
            expert_output = expert_output.reshape(-1, self.hidden_size)
            ans = MoeCombine.apply(expert_output, *route_result_list)
        else:
            combine_weights = route_result_list[0].type_as(inputs)
            combine_weights = combine_weights.view(combine_weights.shape[0], -1)
            expert_output = expert_output.view(-1, expert_output.shape[-1])
            ans = torch.matmul(combine_weights, expert_output)

        # Apply residual connection and reshape back
        ans = ans.reshape(inputs.shape) + residual

        # Return with early exit metadata if requested
        if return_metrics:
            metadata = {
                'early_exit': False,
                'confidence_score': confidence_score.item() if confidence_score is not None else None,
                'complexity_score': complexity_score.item() if complexity_score is not None else None,
            }
            return ans, metadata

        return ans

    def _local_process(self, expert_in: torch.Tensor) -> torch.Tensor:
        expert_in = expert_in.unsqueeze(0)
        expert_out = self.experts(expert_in)
        return expert_out

    def _ep_process(
            self,
            dispatch_data: torch.Tensor,
            used_capacity: torch.Tensor,
            overlap: bool = False
    ) -> torch.Tensor:
        """
        Expert Parallel

        Args:
            dispatch_data (torch.Tensor): (num_experts, capacity, hidden_size)

        Returns:
            torch.Tensor: (num_experts, capacity, hidden_size)
        """
        if not overlap or dist.get_world_size(self.ep_group) == 1:
            if self.ep_hierarchical_group is not None:
                expert_input = HierarchicalAllToAll.apply(dispatch_data, self.ep_hierarchical_group,
                                                          self.ep_intra_src_rank)
                expert_input = expert_input.reshape(self.ep_size, self.num_local_experts, -1, self.hidden_size)
                expert_output = self.experts(expert_input)
                expert_output = HierarchicalAllToAll.apply(expert_output, self.ep_hierarchical_group,
                                                           self.ep_intra_src_rank)
                return expert_output
            else:
                expert_input = AllToAll.apply(dispatch_data, self.ep_group, False)[0]
                expert_input = expert_input.reshape(self.ep_size, self.num_local_experts, -1, self.hidden_size)
                expert_output = self.experts(expert_input)
                expert_output = AllToAll.apply(expert_output, self.ep_group, False)[0]
                return expert_output
        else:

            @dataclasses.dataclass
            class Capsule:
                data: torch.Tensor
                handle: Any = None

            NUM_CHUNK = 4
            NUM_STAGES = 4

            assert (dispatch_data.shape[1] % NUM_CHUNK == 0), "arbitrary chunk num is not supported yet"
            chunk_size = dispatch_data.shape[1] // NUM_CHUNK
            input_shape = (self.ep_size, self.num_local_experts, -1, self.hidden_size)
            dispatch_data = dispatch_data.reshape(*input_shape)
            chunk_data = torch.split(dispatch_data, chunk_size, dim=2)
            output = torch.empty_like(dispatch_data)

            offset = 0
            _expert_in, expert_in, _expert_out, expert_out = None, None, None, None

            for i in range(NUM_CHUNK + NUM_STAGES - 1):
                if expert_out is not None:
                    expert_out.handle.wait()
                    output[:, :, offset:offset + chunk_size, :] = expert_out.data
                    offset += chunk_size
                    expert_out = None

                # all2all last output
                if _expert_out is not None:
                    expert_out = Capsule(*AllToAll.apply(_expert_out.data, self.ep_group, True), )
                    _expert_out = None

                # all2all next input
                if 0 <= i < NUM_CHUNK:
                    _expert_in = Capsule(*AllToAll.apply(chunk_data[i].contiguous(), self.ep_group, True))

                # compute
                if expert_in is not None:
                    expert_in.handle.wait()
                    _expert_out = Capsule(data=self.experts(expert_in.data), handle=None)
                    expert_in = None

                if _expert_in is not None:
                    expert_in = _expert_in
                    _expert_in = None

            return output

    def _tp_process(
            self,
            dispatch_data: torch.Tensor,
            used_capacity: torch.Tensor,
            overlap: bool = False
    ) -> torch.Tensor:
        """
        without overlap:
                   |    C    |
        |     A    |         |    R    |

        with overlap:
              |    C1   ||    C2   ||    C3   ||    C4   |
        | A1 || A2 |     | R1 | A3 || R2 | A4 || R3 |     | R4 |

        where C is computation, A is all gather, R is reduce scatter.

        Args:
            dispatch_data (torch.Tensor): (num_experts, capacity, hidden_size)

        Returns:
            torch.Tensor: (num_experts, capacity, hidden_size)
        """
        if not overlap or dist.get_world_size(self.ep_group) == 1:
            expert_in = AllGather.apply(dispatch_data, self.ep_group, False)[0]
            expert_out = self.experts(expert_in)
            expert_out = ReduceScatter.apply(expert_out, self.ep_group, False)[0]
            return expert_out
        else:

            @dataclasses.dataclass
            class Capsule:
                data: torch.Tensor
                handle: Any
                indices: Tuple

            NUM_CHUNK = 4
            NUM_STAGES = 4

            assert dispatch_data.shape[0] % NUM_CHUNK == 0, \
                "arbitrary chunk num is not supported yet, please use chunk num that can divide num_experts"
            chunk_size = dispatch_data.shape[0] // NUM_CHUNK
            chunk_data = torch.split(dispatch_data, chunk_size, dim=0)
            output = torch.empty_like(dispatch_data)

            def get_chunk_slice(idx: int, chunk_size: int) -> Tuple[slice]:
                return (slice(idx * chunk_size, (idx + 1) * chunk_size),)

            _expert_in, expert_in, _expert_out, expert_out = None, None, None, None

            for i in range(NUM_CHUNK + NUM_STAGES - 1):
                if expert_out is not None:
                    expert_out.handle.wait()
                    output[expert_out.indices] = expert_out.data
                    expert_out = None

                # reduce scatter last output
                if _expert_out is not None:
                    expert_out = Capsule(
                        *ReduceScatter.apply(_expert_out.data, self.ep_group, True),
                        indices=_expert_out.indices,
                    )
                    _expert_out = None

                # all gather next input
                if 0 <= i < NUM_CHUNK:
                    _expert_in = Capsule(
                        *AllGather.apply(chunk_data[i].contiguous(), self.ep_group, True),
                        indices=get_chunk_slice(i, chunk_size),
                    )

                # compute
                if expert_in is not None:
                    expert_in.handle.wait()
                    _expert_out = Capsule(
                        self.experts(expert_in.data, expert_in.indices),
                        handle=None,
                        indices=expert_in.indices,
                    )
                    expert_in = None

                if _expert_in is not None:
                    expert_in = _expert_in
                    _expert_in = None

            return output

    def get_expert_utilization(self):
        """Get expert utilization statistics."""
        if self.enable_layerskip:
            return self.expert_utilization.cpu().numpy()
        return None

    def get_early_exit_stats(self):
        """Get early exit statistics."""
        if self.enable_layerskip:
            stats = self.early_exit_stats.copy()
            if stats['total_tokens'] > 0:
                stats['exit_rate'] = stats['early_exits'] / stats['total_tokens']
            else:
                stats['exit_rate'] = 0.0
            if stats['confidence_scores']:
                stats['avg_confidence'] = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
            else:
                stats['avg_confidence'] = 0.0
            if stats['complexity_scores']:
                stats['avg_complexity'] = sum(stats['complexity_scores']) / len(stats['complexity_scores'])
            else:
                stats['avg_complexity'] = 0.0
            return stats
        return None

    def compute_load_balancing_loss(self):
        """Compute a load balancing loss to encourage uniform expert utilization."""
        if not self.enable_layerskip:
            return 0.0

        # Normalize utilization to sum to 1
        normalized_util = self.expert_utilization / (self.expert_utilization.sum() + 1e-5)
        # Ideal uniform distribution
        uniform_util = torch.ones_like(normalized_util) / self.num_experts
        # KL divergence from uniform
        loss = F.kl_div(normalized_util.log(), uniform_util, reduction='sum')
        return loss

    def train(self, mode=True):
        super().train(mode)
        self.training_mode = mode
        return self

    def eval(self):
        super().eval()
        self.training_mode = False
        return self


def apply_load_balance(model: nn.Module, optim: Any) -> None:
    """
    apply load balance to every experts in the model
    """

    def _apply_recursive(module: nn.Module):
        for _, sub_module in module.named_children():
            if isinstance(sub_module, (SparseMLP, SparseMLPWithLayerSkip)):
                if sub_module.enable_load_balance == True:
                    sub_module.load_balancer.balance_load(optim)
            _apply_recursive(sub_module)

    torch.cuda.empty_cache()
    _apply_recursive(model)
    torch.cuda.empty_cache()


# Add a helper function to compute auxiliary losses for LayerSkip
def compute_layerskip_loss(model: nn.Module, aux_outputs: list, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute auxiliary loss for LayerSkip early exits

    Args:
        model (nn.Module): The model with MoE layers
        aux_outputs (list): List of auxiliary outputs from intermediate layers
        targets (torch.Tensor): Target labels or values

    Returns:
        torch.Tensor: The computed auxiliary loss
    """
    loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # Compute loss for each auxiliary output
    for i, aux_output in enumerate(aux_outputs):
        # Weight earlier layers less
        layer_weight = (i + 1) / len(aux_outputs)
        aux_loss = criterion(aux_output, targets)
        loss += layer_weight * aux_loss

    # Add load balancing loss
    def _collect_moe_modules(module, moe_modules):
        if isinstance(module, SparseMLPWithLayerSkip) and module.enable_layerskip:
            moe_modules.append(module)
        for child in module.children():
            _collect_moe_modules(child, moe_modules)

    moe_modules = []
    _collect_moe_modules(model, moe_modules)

    for moe_module in moe_modules:
        loss += moe_module.compute_load_balancing_loss() * 0.1  # Scaling factor for load balance

    return loss