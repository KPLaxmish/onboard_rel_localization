/*
 * network.h
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "pulp.h"
#include <stddef.h>


void network_terminate();
void network_initialize();
void network_run_cluster(void * args);
void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec);
void execute_layer_fork(void *arg);


#ifdef DEFINE_CONSTANTS
// allocation of buffers with parameters needed by the network execution
static const char * L3_weights_files[] = {
  "BNReluConvolution0_weights.hex", "BNReluConvolution2_weights.hex", "BNReluConvolution4_weights.hex"
};
static int L3_weights_size[3];
static int layers_pointers[5];
static char * Layers_name[5] = {"BNReluConvolution0", "Pooling1", "BNReluConvolution2", "Pooling3", "BNReluConvolution4"};
static int L3_input_layers[5] = {1,
0, 0, 0, 0};
static int L3_output_layers[5] = {0, 0, 0, 0, 0};
static int allocate_layer[5] = {1, 0, 1, 0, 1};
static int branch_input[5] = {0, 0, 0, 0, 0};
static int branch_output[5] = {0, 0, 0, 0, 0};
static int branch_change[5] = {0, 0, 0, 0, 0};
static int weights_checksum[5] = {8287, 0, 46456, 0, 20527};
static int weights_size[5] = {68, 0, 352, 0, 160};
static int activations_checksum[5][1] = {{
  7223661  },
{
  649109  },
{
  218575  },
{
  678159  },
{
  231255  }
};
static int activations_size[5] = {102400, 102400, 25600, 51200, 12800};
static int out_mult_vector[5] = {1, 1, 1, 1, 1};
static int out_shift_vector[5] = {22, 0, 22, 0, 22};
static int activations_out_checksum[5][1] = {{
  649109 },
{
  218575 },
{
  678159 },
{
  231255 },
{
  20291 }
};
static int activations_out_size[5] = {102400, 25600, 51200, 12800, 3200};
static int layer_with_weights[5] = {1, 0, 1, 0, 1};
#endif

#endif  // __NETWORK_H__
