# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import torch

from amct_pytorch.utils.quant_util import convert_dtype, apply_smooth_weight


class NpuQuantizationLinear(torch.nn.Module):
    """
    Function: Customized torch.nn.Module of the npu quantized linear operator.
    APIs: forward
    """
    def __init__(self, quant_module):
        super().__init__()
        self.output_dtype = quant_module.weight.dtype
        device = quant_module.weight.device
        self.offset_bias = None

        self.act_type = quant_module.act_type
        self.act_granularity = 'PER_TENSOR'
        self.quantize_axis = -1
        scale_d_tensor = quant_module.scale_d.to(device=device)
        offset_d_tensor = quant_module.offset_d if quant_module.offset_d is not None else torch.tensor([0])
        offset_d_tensor = offset_d_tensor.repeat(scale_d_tensor.shape)
        if scale_d_tensor.dim() > 1:
            self.act_granularity = 'PER_TOKEN'
            self.quantize_axis = -2
        self.register_buffer('act_scale', 
                             (1 / scale_d_tensor).reshape(-1).to(device=device).to(self.output_dtype))
        self.register_buffer('act_offset', 
                             offset_d_tensor.reshape(-1).to(device=device).to(self.output_dtype))

        wts_type = quant_module.wts_type
        weight = quant_module.weight.data
        if quant_module.scale is not None:
            weight = apply_smooth_weight(quant_module.scale, weight)
            self.register_buffer('scale_factor', (1 / quant_module.scale).to(device=device))
        scale_w_tensor = quant_module.scale_w.reshape(-1).to(device=device)
        weight_tensor = NpuQuantizationLinear.quant_wts(weight, scale_w_tensor, wts_type)
        weight_tensor = torch.transpose(weight_tensor, 1, 0)
        self.register_buffer('quantized_weight', weight_tensor)
        if quant_module.offset_d is not None:
            offset_reshape = quant_module.offset_d.float().repeat(1, quant_module.weight.shape[-1])
            self.offset_bias = offset_reshape.float() @ weight_tensor.float()
            self.offset_bias = -1 * self.offset_bias.round().to(torch.int32).reshape(-1)

        deq_scale_tensor = quant_module.scale_d.to(device=device) * scale_w_tensor
        if self.act_granularity == 'PER_TENSOR':
            self.register_buffer('deq_scale', deq_scale_tensor)
            self.pertoken_scale = None
        else:
            self.register_buffer('deq_scale', scale_w_tensor)
            self.register_buffer('pertoken_scale',
                                 quant_module.scale_d.reshape(-1).to(torch.float32).to(device=device))

        self._init_bias(quant_module, scale_d_tensor, scale_w_tensor)

    @staticmethod
    def quant_wts(weight, scale_w_tensor, wts_type):
        """
        Function: weight quant.
        Params:
            weight: the original or scaled weight to do quantization
            scale_w_tensor: torch.tensor, scale factor for weight
            wts_type: weight quantized data type
        Returns:
            torch.tensor
        """
        if scale_w_tensor.shape[0] > 1:
            scale_w_tensor = scale_w_tensor.reshape(-1, 1)
        weight = weight / scale_w_tensor
        quant_weight = convert_dtype(weight, wts_type)
        return quant_weight

    def forward(self, x):
        """
        do inference process
        Params:
        inputs: input data in torch.tensor.
        """
        import torch_npu
        if hasattr(self, 'scale_factor'):
            x = x * self.scale_factor
        quant_x = torch_npu.npu_quantize(x, self.act_scale, self.act_offset, dtype=torch.int8, 
            axis=self.quantize_axis, div_mode=False)
        output = torch_npu.npu_quant_matmul(quant_x, self.quantized_weight,
            scale=self.deq_scale, pertoken_scale=self.pertoken_scale,
            bias=self.bias, output_dtype=self.output_dtype)
        return output

    def _init_bias(self, module, scale_d_tensor, scale_w_tensor):
        """
        Function: init bias for npu op
        Args:
            module: quant module 
            scale_d_tensor: torch.tensor, scale factor for activation
            scale_w_tensor: torch.tensor, scale factor for weight
        Returns:
            torch.tensor
        """
        if module.bias is not None:
            bias = module.bias
            if self.act_granularity == 'PER_TENSOR':
                bias_tensor = bias.data / (scale_d_tensor * scale_w_tensor)
                bias_tensor = bias_tensor.round().to(torch.int32)
                self.register_buffer('bias', bias_tensor)
            elif self.act_granularity == 'PER_TOKEN':
                self.register_buffer('bias', bias)

            if self.offset_bias is not None:
                self.bias = self.bias + self.offset_bias
        else:
            if self.offset_bias is None:
                self.bias = None 
            else:
                self.bias = self.offset_bias

