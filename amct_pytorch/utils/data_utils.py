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

DATA_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2
}


def check_linear_input_dim(input_tensor):
    """
    Function: Check if the input dimension for linear operation is between 2 and 6
    Args:
        input_tensor: Input tensor to check dimension
    """
    input_dim = len(input_tensor.shape)
    if input_dim < 2 or input_dim > 6:
        raise RuntimeError("Linear quant only support dim from 2 to 6")


def check_data_type(tensor_dtype, data_types):
    """
    Function: check tensor dtype in data_types
    Args:
        tensor_dtype: torch.dtype
        data_types: list or tuple. torch.dtypes
    """
    if tensor_dtype not in data_types:
        raise RuntimeError('Not support tensor dtype {}, support dtypes {}.'.format(tensor_dtype, data_types))


@torch.no_grad()
def convert_precision(ori_tensor, quant_dtype):
    """
    Function: convert precision to quant_dtype and back.
    Args:
        ori_tensor: torch.tensor
        quant_dtype: quant type
    Returns:
        torch.tensor
    """
    original_dtype_index = DATA_MAP.get(ori_tensor.dtype)
    if original_dtype_index is None:
        raise RuntimeError(
            "dtype {} not support now, only support float32/float16/bfloat16.".format(ori_tensor.dtype))

    quant_bits = int(quant_dtype.replace('int', ''))
    converted_data = torch.clamp(torch.round(ori_tensor), -pow(2, quant_bits - 1), pow(2, quant_bits - 1) - 1)

    return converted_data
