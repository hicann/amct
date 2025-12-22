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

import math
import torch

from amct_pytorch.utils.quant_util import pad_zero_by_group, convert_to_per_group_shape

FLT_EPSILON = 1.192092896e-7


def process_scale(scale, offset, symmetric, numbit=8):
    """
    Function: check the validity of quant factor(scale)'s range
    Inputs:
        scale: torch.Tensor, quant factor
        offset: torch.Tensor, quant factor
        with_offset: bool, with offset or not
        numbit: int, quant bits
    Outputs:
        scale: torch.Tensor, quant factor
        offset: torch.Tensor, quant factor
    """
    base_bit = 2.0
    if symmetric:
        scale = torch.where(
            scale < FLT_EPSILON,
            torch.tensor(1.0, device=scale.device), scale)
    else:
        offset = torch.where(
            scale < FLT_EPSILON,
            torch.tensor(-pow(base_bit, numbit - 1), device=offset.device), offset)
        scale = torch.where(
            scale < FLT_EPSILON,
            torch.tensor(1.0, device=scale.device), scale)
    check_scale_offset(scale, offset)
    return scale, offset


def check_scale_offset(scale, offset):
    """
    Function: check scale and offset for ops
    Inputs:
        scale: torch.Tensor
        offset: torch.Tensor
    """
    valid = True
    if not torch.all(torch.isfinite(scale)):
        valid = False
    if not torch.all(torch.isfinite(1 / scale)):
        valid = False
    if not valid:
        raise RuntimeError(
            "Param scale has invalid value inf, nan or zeros!")

    if offset is not None and not torch.all(torch.isfinite(offset)):
        raise RuntimeError(
            "Param offset has invalid value inf or nan!")


def calculate_scale_offset(data_max, data_min, symmetric, data_type):
    """
    Function: calculate scale and offset using min-max algorithms
    Parameters: data_max: max value in calibration data
                data_min: min value in calibration data
                symmetric: do symmetric quant
                data_type: Type of integer data (e.g., INT8, INT4)
    Return: scale: scale in quant factors
            offset: offset in quant factors
    """
    data_max = data_max.to(torch.float32)
    data_min = data_min.to(torch.float32)
    if symmetric:
        abs_max_is_negative = data_max.abs() < data_min.abs()
        quant_scope = get_int_quant_scope(data_type, symmetric, abs_max_is_negative)
        boundary = torch.where(abs(data_max) > abs(data_min), abs(data_max), abs(data_min))
        scale = boundary / quant_scope
        offset = None
    else:
        # clamp lower bound of data_max to 0
        data_max = torch.clamp(data_max, 0, None)
        # clamp upper bound of data_max to 0
        data_min = torch.clamp(data_min, None, 0)
        quant_scope = get_int_quant_scope(data_type, symmetric)

        scale = (data_max - data_min) / quant_scope
        offset = torch.where(data_max != data_min,
            -math.ceil(quant_scope / 2) - data_min / scale,
            torch.tensor(-math.ceil(quant_scope / 2), dtype=data_max.dtype, device=data_max.device))
    int_quant_bits = int(data_type.replace('int', '')) if 'int' in data_type else None
    scale, offset = process_scale(scale, offset, symmetric, int_quant_bits)
    scale = scale.to(torch.float32)
    offset = None if offset is None else offset.round().to(torch.int32)
    return scale, offset


def get_weight_min_max_by_granularity(weight_data, quant_config):
    """
    Calculate the minimum and maximum values of weights based on the module type and weight data.
    The calculation method depends on the granularity setting in the quantization configuration.

    Parameters:
    weight_data (Tensor): Weight data
    quant_config (dict): Quantization configuration

    Returns:
    tuple: A tuple containing the minimum and maximum values of the weights
    """
    if quant_config.get('weights_cfg').get('strategy') == 'channel':
        weight_max = weight_data.max(dim=1, keepdim=True).values
        weight_min = weight_data.min(dim=1, keepdim=True).values
    elif quant_config.get('weights_cfg').get('strategy') == 'tensor':
        weight_max = weight_data.max().reshape(1, 1)
        weight_min = weight_data.min().reshape(1, 1)
    else:
        # weight: [n, k] -> [cout, cin//group_size, group_size]
        group_size = quant_config.get('weights_cfg').get("group_size")
        weight_all = convert_to_per_group_shape(weight_data, group_size)
        weight_max = weight_all.max(dim=-1, keepdim=True).values  # [cout, cin//group_size, 1]
        weight_min = weight_all.min(dim=-1, keepdim=True).values  # [cout, cin//group_size, 1]
 
    return weight_min, weight_max


def convert_to_dst_shape(input_tensor, dst_shape):
    """
    Converts a flattened tensor back to its destination 2D shape by truncating excess elements.
 
    Parameters:
        input_tensor (Tensor): Input tensor to be reshaped (can be padded or extended)
        dst_shape (tuple): destination 2D shape (rows, columns) to restore
 
    Returns:
        Tensor: Reshaped tensor matching the destination dimensions
    """
    if input_tensor.shape == dst_shape:
        return input_tensor

    out_tensor = input_tensor.reshape(input_tensor.shape[0], \
        input_tensor.shape[1] * input_tensor.shape[2])[:dst_shape[0], :dst_shape[1]]
    return out_tensor


def get_int_quant_scope(data_type, symmetric, abs_max_is_negative=False):
    """
    Get the quantization step for integer data types.

    Args:
        data_type: Type of integer data (e.g., INT8, INT4)
        symmetric: True if using symmetric quantization, False for asymmetric
        abs_max_is_negative: The absolute value of the maximum value of the original data is negative

    Returns:
        quantization step
    """
    quant_bits = int(data_type.replace('int', ''))
    if not symmetric:
        return pow(2, quant_bits) - 1

    # Symmetric quantization, asymmetric quantization range (-128, 127), use 127 for positive values and
    # 128 for negative values when the absolute value of the original data is the maximum. 
    return abs_max_is_negative + pow(2, quant_bits - 1) - 1
