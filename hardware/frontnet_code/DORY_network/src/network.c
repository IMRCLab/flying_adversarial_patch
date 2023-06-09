/*
 * network.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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
#define DEFINE_CONSTANTS
#include "network.h"
#include "dory.h"
#include "BNReluConvolution0.h"
#include "BNReluConvolution3.h"
#include "BNReluConvolution4.h"
#include "FullyConnected8.h"
#include "BNReluConvolution5.h"
#include "BNReluConvolution7.h"
#include "BNReluConvolution6.h"
#include "Pooling1.h"
#include "BNReluConvolution2.h"

#define FLASH_BUFF_SIZE 128
#define VERBOSE 1

static uint8_t flashBuffer[FLASH_BUFF_SIZE];
int memId;
char* L2_output;
char* L2_input;
char* L2_weights;
char* l1_buffer;
char* bypass_activations;
int L3_weights_internal;

#ifdef VERBOSE
// check for input/output acitvation checksum
static void check_layer(uint8_t *output, int check_sum_true, int dim) {
  int checksum = 0;
  uint8_t *ptr = (uint8_t *) output;
  for(int j=0; j<dim; j++) {
    checksum += *(uint8_t *)(ptr+j);
  }

  if(check_sum_true == checksum)
    printf("Checksum in/out Layer :\tOk\n");
  else
    printf("Checksum in/out Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);
}

static void check_layer_last(int32_t *ptr, int check_sum_true, int dim) {
  int checksum = 0;
  for(int j=0; j<dim/4; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum final :\tOk\n");
  else
    printf("Checksum final :\tFailed [%d vs. %d]\n", checksum, check_sum_true);
}

// check for weight checksum
static void check_layer_weight(char *weight, int check_sum_true, int dim) {
  int checksum = 0;
  char *ptr = (char *) weight;
  for(int j=0; j<dim; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum weight/bias Layer :\tOk\n");
  else
    printf("Checksum weight/bias Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);
}
#endif

/* Moves the weights and the biases from hyperflash to hyperram */
void network_alloc(struct pi_device fs, struct pi_device ram)
{
  pi_fs_file_t *file;
  pi_ram_alloc(&ram, &L3_weights, (uint32_t) 4500000);
  pi_ram_alloc(&ram, &L3_input, (uint32_t) 1500000);
  pi_ram_alloc(&ram, &L3_output, (uint32_t) 1500000);
#ifdef VERBOSE
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_weights, L3_weights?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_input, L3_input?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_output, L3_output?"Ok":"Failed");
#endif
  unsigned int rdDone = 0;
  int layer_number = 0;
  int sum_weights;
  for (int i=0;i<8;i++)
  {
    if (layer_with_weights[layer_number]==0)
      layer_number +=1;
    file = pi_fs_open(&fs, L3_weights_files[i], 0);
    if (file == NULL)
    {
      printf("file %s open failed \n", L3_weights_files[i]);
      return -1;
    }
    L3_weights_size[i] = file->size + rdDone;
    int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
    sum_weights = 0;
    while(rdDone < (L3_weights_size[i] / sizeof(char)))
    {
      int size = pi_fs_read(file, flashBuffer, flashBuffSize);
      for (int t = 0; t < size; t++)
        sum_weights+=flashBuffer[t];
      pi_ram_write(&ram, L3_weights+rdDone, flashBuffer,size);
      rdDone += size / sizeof(char);
    }
    if (check_weights[layer_number] == sum_weights)
      printf("Layer %-3d: Checksum = %-12d, FLASH %-12d, Check OK\n", layer_number, check_weights[layer_number], sum_weights);
    else
      printf("Layer %-3d: Checksum = %-12d, FLASH %-12d, Check FAILED\n", layer_number, check_weights[layer_number], sum_weights);
    layer_number +=1;
  }
  return 1;
}

/* Remove RAM memory */
void network_free(struct pi_device ram)
{
  pi_ram_free(&ram, L3_weights, (uint32_t) 4500000);
  pi_ram_free(&ram, L3_input, (uint32_t) 1500000);
  pi_ram_free(&ram, L3_output, (uint32_t) 1500000);
}

void execute_layer_fork(void *arg)
{
  unsigned int *real_arg = (unsigned int *) arg;
  real_arg[7] = pmsis_l1_malloc((uint32_t) 36700);
  void *args = (void *) real_arg;
  switch (real_arg[11])
  {
    case 0:
      pi_cl_team_fork(NUM_CORES, (void *)BNReluConvolution0, args);
      break;
    case 1:
      pi_cl_team_fork(NUM_CORES, (void *)Pooling1, args);
      break;
    case 2:
      pi_cl_team_fork(NUM_CORES, (void *)BNReluConvolution2, args);
      break;
    case 3:
      pi_cl_team_fork(NUM_CORES, (void *)BNReluConvolution3, args);
      break;
    case 4:
      pi_cl_team_fork(NUM_CORES, (void *)BNReluConvolution4, args);
      break;
    case 5:
      pi_cl_team_fork(NUM_CORES, (void *)BNReluConvolution5, args);
      break;
    case 6:
      pi_cl_team_fork(NUM_CORES, (void *)BNReluConvolution6, args);
      break;
    case 7:
      pi_cl_team_fork(NUM_CORES, (void *)BNReluConvolution7, args);
      break;
    case 8:
      pi_cl_team_fork(NUM_CORES, (void *)FullyConnected8, args);
      break;
  }
  if (pi_core_id()==0)
    pmsis_l1_malloc_free(real_arg[7], (uint32_t) 36700);
}

void network_run(char *L2_memory_buffer, int L2_memory_dimension, char *L2_output_to_pass, int begin_end, struct pi_device ram)
{

/*
  - initial buffer allocation L2 and L1
  - variable declaration
*/
/* ---------------------------------- */
/* -------- SECTION 0 BEGIN --------- */
/* ---------------------------------- */
  bypass_activations = 0;
  int begin_end_n = begin_end;
  int L2_memory_buffer_end = L2_memory_buffer + L2_memory_dimension;
  int residual_number = 0;
  L3_weights_internal = L3_weights;
  int perf_cyc = 0;
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;

/* ---------------------------------- */
/* --------- SECTION 0 END ---------- */
/* ---------------------------------- */

/*
  - initial copies from L3 of input
  - copies of weights of first 2 layers
*/
/* ---------------------------------- */
/* -------- SECTION 1 BEGIN --------- */
/* ---------------------------------- */

/*
  - input allocation and copy
*/
  dory_L2_alloc(&L2_memory_buffer,
    &L2_memory_buffer_end,
    &L2_input,
    check_activations_dimension[0],
    begin_end_n // begin is 1, end is 0
    );
  begin_end_n = !begin_end_n;
/*
  - output of the first layer allocation
*/
  dory_L2_alloc(&L2_memory_buffer,
    &L2_memory_buffer_end,
    &L2_output,
    check_activations_out_dimension[0],
    begin_end_n // begin is 1, end is 0
    );
  if(L2_output == NULL) return -1;
  begin_end_n = !begin_end_n;
/* ---------------------------------- */
/* --------- SECTION 1 END ---------- */
/* ---------------------------------- */
  // perf measurement begin
  int cycle_network_execution = 0;
/* MAIN SECTION
  - for loop over all the layers of the network
  - double buffering using L3
  - check on layers to be executed from L3
  - residual check at the end of each layer
*/
/* ---------------------------------- */
/* -------- SECTION 2 BEGIN --------- */
/* ---------------------------------- */
  for(int i = 0; i < 9; i++)
  {
/* MEMORY ALLOCATION
  - allocate memory if layer is executed from L3;
  - allocate weights
  - read weights
*/
    if (L3_input_layers[i] == 1)
      dory_L2_alloc(&L2_memory_buffer,
        &L2_memory_buffer_end,
        &L2_input,
        check_activations_dimension[i],
        begin_end_n // begin is 1, end is 0
        );
    if (layer_with_weights[i] == 1)
      dory_L2_alloc(&L2_memory_buffer,
        &L2_memory_buffer_end,
        &L2_weights,
        check_weights_dimension[i],
        begin_end_n // begin is 1, end is 0
        );
    if (allocate_layer[i] == 1)
    {
      pi_ram_read(&ram, L3_weights_internal + cumulative_weights_dimension[i], L2_weights, check_weights_dimension[i]);
    }

#ifdef VERBOSE
    if (L3_input_layers[i]==1)
      printf("Input in L3\n");
    else if (i==0) {
      printf("Checking input of layer %d...\n", i);
      check_layer(L2_input, check_activations[i], check_activations_dimension[i]);
      if (allocate_layer[i] == 1)
      {
        check_layer_weight(L2_weights, check_weights[i], check_weights_dimension[i]);
      }
      else
      {
        printf("Weights in L3\n");
      }
    }
    else if (branch_change[i-1]==0) {
      printf("Checking input of layer %d...\n", i);
      check_layer(L2_input, check_activations[i], check_activations_dimension[i]);
      if (allocate_layer[i] == 1)
      {
        check_layer_weight(L2_weights, check_weights[i], check_weights_dimension[i]);
      }
      else
      {
        printf("Weights in L3\n");
      }
    }
    else
      printf("Switching branch, already checked activation\n");
#endif
    unsigned int args[12] = {L3_input,
      L3_output,
      L3_weights_internal + cumulative_weights_dimension[i],
      L2_input,
      bypass_activations,
      L2_output,
      L2_weights,
      l1_buffer,
      &ram,
      out_mult_vector[i],
      out_shift_vector[i],
      i};

/*
- Execution of the layers_pointers
*/
    // perf measurement begin
    pi_perf_conf(1<<PI_PERF_CYCLES);
    pi_perf_reset();
    pi_perf_stop();
    pi_perf_start();
    pi_cluster_task(&cluster_task, execute_layer_fork, args);
    pi_open_from_conf(&cluster_dev, &conf);
    if (pi_cluster_open(&cluster_dev)) {
      printf("OPEN FAILED");
      return -1;
    }
    // Then offload an entry point, this will get executed on the cluster controller
    cluster_task.stack_size = 3500;
    cluster_task.slave_stack_size = 3400;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    // closing of the cluster
    pi_cluster_close(&cluster_dev);
    // performance measurements: end
    pi_perf_stop();
    perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
    cycle_network_execution += perf_cyc;
    

    // prevents error from compiler
    asm volatile("": : :"memory");
    unsigned int temp = L3_input;
    L3_input = L3_output;
    asm volatile("": : :"memory");
    L3_output = temp;
    asm volatile("": : :"memory");

#ifdef VERBOSE
    printf("Layer %s %d ended: \n", Layers_name[i], i);
    if (i < 8)
    {
      if (L3_output_layers[i]==1)
        printf("Out in L3\n");
      else {
        check_layer(L2_output, check_activations_out[i], check_activations_out_dimension[i]);
      }
      printf("\n");
    }
    else
    {
      check_layer_last((int32_t *) L2_output, check_activations_out[i], check_activations_out_dimension[i]);
    }
#endif

/* MEMORY DEALLOCATION
*/
    // deallocation of weights
    if (layer_with_weights[i] == 1)
      dory_L2_free(&L2_memory_buffer,
        &L2_memory_buffer_end,
        check_weights_dimension[i],
        begin_end_n // begin is 1, end is 0
        );
    dory_L2_free(&L2_memory_buffer,
      &L2_memory_buffer_end,
      check_activations_dimension[i],
      begin_end_n // begin is 1, end is 0
      );
    if (branch_input[i]==1)
      dory_L2_free(&L2_memory_buffer,
        &L2_memory_buffer_end,
        check_activations_dimension[i],
        begin_end_n // begin is 1, end is 0
        );
    L2_input = L2_output;
    if (i < 8)
    {
      if (branch_input[i+1]==1)
      {
        begin_end_n = !begin_end_n;
        dory_L2_alloc(&L2_memory_buffer,
          &L2_memory_buffer_end,
          &bypass_activations,
          check_activations_out_dimension[i],
          begin_end_n // begin is 1, end is 0
          );
        begin_end_n = !begin_end_n;
        residual_number--;
        pi_ram_read(&ram, layers_pointers[residual_number], bypass_activations, check_activations_out_dimension[i]);
        pi_ram_free(&ram, layers_pointers[residual_number], (uint32_t) check_activations_out_dimension[i]);
      }
      if (i>0)
      {
        if (branch_output[i-1]==1 && L3_input_layers[i]==1)
        {
          pi_ram_alloc(&ram, &L3_input, (uint32_t) 1500000);
        }
      }
      if (branch_output[i]==1 && L3_output_layers[i]==1)
      {
        pi_ram_free(&ram, (uint32_t) L3_input + check_activations_out_dimension[i], (uint32_t) 1500000 - check_activations_out_dimension[i]);
        layers_pointers[residual_number] = L3_input;
        residual_number++;
      }
      else if (branch_output[i]==1 || branch_change[i] == 1)
      {
        int32_t temp_adress;
        pi_ram_alloc(&ram, &temp_adress, (uint32_t) check_activations_out_dimension[i]);
        layers_pointers[residual_number] = temp_adress;
        pi_ram_write(&ram, temp_adress, L2_output, (uint32_t) check_activations_out_dimension[i]);
        residual_number++;
      }
      if (branch_change[i]==1)
      {
        begin_end_n = !begin_end_n;
        dory_L2_free(&L2_memory_buffer,
          &L2_memory_buffer_end,
          check_activations_out_dimension[i],
          begin_end_n // begin is 1, end is 0
          );
        dory_L2_alloc(&L2_memory_buffer,
          &L2_memory_buffer_end,
          &L2_input,
          check_activations_dimension[i+1],
          begin_end_n // begin is 1, end is 0
          );
        begin_end_n = !begin_end_n;
        residual_number--;
        residual_number--;
        pi_ram_read(&ram, layers_pointers[residual_number], L2_input, check_activations_dimension[i+1]);
        pi_ram_free(&ram, layers_pointers[residual_number], (uint32_t) check_activations_out_dimension[i+1]);
        residual_number++;
        residual_number++;
      }
      dory_L2_alloc(&L2_memory_buffer,
        &L2_memory_buffer_end,
        &L2_output,
        check_activations_out_dimension[i+1],
        begin_end_n // begin is 1, end is 0
        );
      //switching output and input in the buffer for allocation.
      begin_end_n = !begin_end_n;
      if (L3_output_layers[i] == 1)
        dory_L2_free(&L2_memory_buffer,
          &L2_memory_buffer_end,
          check_activations_out_dimension[i],
          begin_end_n // begin is 1, end is 0
          );
    }
  }
  for(int i = 0; i < check_activations_out_dimension[9]; i++)
  {
    *(L2_output_to_pass + i) = *(L2_output + i);
  }
/* ---------------------------------- */
/* --------- SECTION 2 END ---------- */
/* ---------------------------------- */

/* ---------------------------------- */
/* -------- SECTION 3 BEGIN --------- */
/* ---------------------------------- */

  int MACs = 14138880;
  float perf_MAC =  (float)MACs/cycle_network_execution;
  printf("\nnum_cycles: %d\n",cycle_network_execution);
  printf("MACs: %d\n",MACs );
  printf("MAC/cycle: %f\n",perf_MAC );
  printf("n. of Cores: %d\n",NUM_CORES);

/* ---------------------------------- */
/* --------- SECTION 3 END ---------- */
/* ---------------------------------- */
}

