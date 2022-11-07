/*
 * layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Francesco Conti <f.conti@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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

#include "BNReluConvolution3.h"

void BNReluConvolution3(
  void *args
) {
  //////////////////////////////////////////////////////////////////////////
  // arguments assigning: keeping same interface between L2 and L3 memory //
  //////////////////////////////////////////////////////////////////////////
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x =(unsigned int)  real_arg[0];
  unsigned int l3_y =(unsigned int)  real_arg[1];
  unsigned int l3_W =(unsigned int)  real_arg[2];
  unsigned int l2_x =(unsigned int)  real_arg[3];
  unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_mult_in =(unsigned int)  real_arg[9];
  unsigned int out_shift_in = (unsigned int) real_arg[10];

  /////////////////////
  // DMA declaration //
  /////////////////////
  uint32_t dory_dma_channel = dory_dma_allocate();
  volatile DMA_copy DMA_copy_k, DMA_copy_lambda;
  volatile DMA_copy DMA_copy_W, DMA_copy_x, DMA_copy_y;
  DMA_copy_k.hwc_to_chw = 0;
  DMA_copy_k.stride_2d = 0;
  DMA_copy_k.stride_1d = 0;
  DMA_copy_k.dir = 1;
  DMA_copy_k.dma_channel = dory_dma_channel;

  DMA_copy_lambda.hwc_to_chw = 0;
  DMA_copy_lambda.stride_2d = 0;
  DMA_copy_lambda.stride_1d = 0;
  DMA_copy_lambda.dir = 1;
  DMA_copy_lambda.dma_channel = dory_dma_channel;
  
  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = 640;
  DMA_copy_x.stride_1d = 32;
  DMA_copy_x.dir = 1;
  DMA_copy_x.dma_channel = dory_dma_channel;
  
  DMA_copy_W.hwc_to_chw = 0;
  DMA_copy_W.stride_2d = 288;
  DMA_copy_W.stride_1d = 32;
  DMA_copy_W.dir = 1;
  DMA_copy_W.dma_channel = dory_dma_channel;
  
  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = 640;
  DMA_copy_y.stride_1d = 32;
  DMA_copy_y.dir = 0;
  DMA_copy_y.dma_channel = dory_dma_channel;

  volatile int p_r, p_l, p_t, p_b;
  volatile unsigned short  W_tile_size_nof;
  volatile unsigned short  W_tile_size_nif;
  volatile unsigned short  W_tile_size_byte;
  volatile unsigned short W_length_nif_byte;
  volatile char *x, *W, *y, *b;
  volatile int64_t *k;
  volatile int64_t *lambda;
  volatile int x_tile_size_nif_exec;
  volatile int x_tile_size_h_exec;
  volatile int x_tile_size_w_exec;
  volatile int y_tile_size_nof;
  volatile int y_tile_size_h;
  volatile int y_tile_size_w;
  volatile int y_tile_size_byte;
  volatile int y_length_nof_byte;
  volatile int db_x;
  volatile int db_W;
  volatile int db_act;
  volatile int db_y;
  volatile int exec_db_x;
  volatile int exec_db_W;
  volatile int exec_db_act;
  // double buffering state
  int db_state_x=0;
  int db_state_W=0;
  int db_state_y=1;
  // last-tile flags
  int iter;
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;
  volatile char *im2col;
  im2col = l1_buffer + 25128;
  uint16_t out_mult = out_mult_in;
  uint16_t out_shift = out_shift_in;

  ////////////////////////////
  // First tile transfering //
  ////////////////////////////
  DMA_copy_k.ext = (uint32_t) l2_W+9216;
  DMA_copy_k.loc = (uint32_t) l1_buffer + 24600;
  DMA_copy_k.number_of_2d_copies = 1;
  DMA_copy_k.number_of_1d_copies = 1;
  DMA_copy_k.length_1d_copy = (uint16_t) 256;
  dory_dma_memcpy_async(DMA_copy_k);
  dory_dma_barrier(DMA_copy_k);

  DMA_copy_lambda.ext = (uint32_t) l2_W+9472;
  DMA_copy_lambda.loc = (uint32_t) l1_buffer + 24864;
  DMA_copy_lambda.number_of_2d_copies = 1;
  DMA_copy_lambda.number_of_1d_copies = 1;
  DMA_copy_lambda.length_1d_copy = (uint16_t) 256;
  dory_dma_memcpy_async(DMA_copy_lambda);
  dory_dma_barrier(DMA_copy_lambda);


  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = (l1_buffer + 0) + 0;
  DMA_copy_x.number_of_2d_copies = 12;
  DMA_copy_x.number_of_1d_copies = 20;
  DMA_copy_x.length_1d_copy = 32;
  dory_dma_memcpy_async(DMA_copy_x);
  dory_dma_barrier(DMA_copy_x);

  DMA_copy_W.ext = l2_W;
  DMA_copy_W.loc = (l1_buffer + 15376) + 0;
  DMA_copy_W.number_of_2d_copies = 32;
  DMA_copy_W.number_of_1d_copies = 9;
  DMA_copy_W.length_1d_copy = 32;
  dory_dma_memcpy_async(DMA_copy_W);
  dory_dma_barrier(DMA_copy_W);

  dory_cores_barrier();

  int total_tiles = 1;
  // tile loop nest
  for(iter=0; iter < total_tiles; iter++) {
      _i_w_load += 1;
      if(_i_w_load==1) 
      {
        _i_w_load = 0;
        _i_h_load += 1;
        if(_i_h_load==1) 
        {
          _i_h_load = 0;
          _i_nof_load += 1;
        }
      }
    // check if last in any dimension

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? 7680 : 0;
    db_W = !db_state_W ? 9216 : 0;
    db_y = !db_state_y ? 7680 : 0;
    db_act = !db_state_W ? 256 : 0;
    exec_db_x = 0;
    db_state_x = ! db_state_x;
    exec_db_W = db_state_W ? 9216 : 0;
    exec_db_act = db_state_W ? 256 : 0;
    if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      db_state_W = ! db_state_W;
    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single fil
///////// POSSIBLE BUG FIX!!!!! DB_STATE_Y NOT SWITCHED /////////////

    // double buffered reads

    if(iter < (total_tiles-1) )
    {
      asm volatile("": : :"memory");
      y_tile_size_h   = (_i_h_load+1 == 1)   ? 12 : 12;
      y_tile_size_w   = (_i_w_load+1 == 1)   ? 20 : 20;
      W_tile_size_nof = (_i_nof_load+1 == 1) ? 32 : 32;
      W_tile_size_nif = (_i_nif_load+1 == 1) ? 32 : 32;
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*8*3*3/8;
      W_length_nif_byte = (_i_nif_load+1 == 1) ? 32 : 32;
      // transfer of next input tile in double buffering
      // transfer of next weight tile if changed input or output channels
      if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      {
        DMA_copy_W.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, 32, 3*3, 32, 3*3, 32, 0,0,0,0,0,0, 8);
        DMA_copy_W.loc = (l1_buffer + 15376) + db_W;
        DMA_copy_W.number_of_2d_copies = W_tile_size_nof;
        DMA_copy_W.length_1d_copy = W_length_nif_byte;
        dory_dma_memcpy_async(DMA_copy_W);

        DMA_copy_k.ext = (uint32_t) l2_W+9216 + 256*_i_nof_load;
        DMA_copy_k.loc = (uint32_t) l1_buffer + 24600 + db_act;
        DMA_copy_k.length_1d_copy = (uint16_t) W_tile_size_nof * 8;
        dory_dma_memcpy_async(DMA_copy_k);

        DMA_copy_lambda.ext = (uint32_t) l2_W+9472 + 256*_i_nof_load;
        DMA_copy_lambda.loc = (uint32_t) l1_buffer + 24864 + db_act;
        DMA_copy_lambda.length_1d_copy = (uint16_t) W_tile_size_nof * 8;
        dory_dma_memcpy_async(DMA_copy_lambda);
      }
    }
    // creation of the pointers to input, output, weights, lambda and k
    x = (char *) (l1_buffer + 0 + exec_db_x);
    k = (int64_t *) (l1_buffer + 24600 + exec_db_act);
    lambda = (int64_t *) (l1_buffer + 24864 + exec_db_act);
    W = (char *) (l1_buffer + 15376 + exec_db_W);
    y = (char *) (l1_buffer + 7688 + db_y);
    // parameter passed to the kernel. Input and output sizes
    x_tile_size_nif_exec = (_i_nif_exec+1 == 1) ? 32 : 32;
    x_tile_size_h_exec   = (_i_h_exec+1 == 1)   ? 12 : 12;
    x_tile_size_w_exec   = (_i_w_exec+1 == 1)   ? 20 : 20;
    y_tile_size_nof = (_i_nof_exec+1 == 1) ? 32 : 32;
    y_tile_size_h   = (_i_h_exec+1 == 1)   ? 12 : 12;
    y_tile_size_w   = (_i_w_exec+1 == 1)   ? 20 : 20;
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*8/8;
    y_length_nof_byte = (_i_nof_exec+1 == 1)   ? 32 : 32;
    p_r = 0;
    p_l = 0;
    p_t = 0;
    p_b = 0;
    if (_i_h_exec == 0)
      p_t = 1;
    if (_i_w_exec == 0)
      p_l = 1;
    if (_i_h_exec == 1-1)
      p_b = 1;
    if (_i_w_exec == 1-1)
      p_r = 1;
    dory_cores_barrier();
    asm volatile("": : :"memory");
    pulp_nn_conv_Ho_parallel(
      x, im2col,
      NULL,
      y, W,
      k, lambda,
      out_mult, out_shift,
      x_tile_size_w_exec, x_tile_size_h_exec, x_tile_size_nif_exec,
      y_tile_size_w, y_tile_size_h, y_tile_size_nof,
      3, 3,
      p_t, p_b, p_l, p_r, 1, 1,
      1, 1
      );
    dory_cores_barrier();
    // wait for DMA write/read
      dory_dma_barrier(DMA_copy_y);
      dory_dma_barrier(DMA_copy_x);
      dory_dma_barrier(DMA_copy_W);

    if(iter < (total_tiles-1) && (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec))
    {                        
      dory_dma_barrier(DMA_copy_k);
      dory_dma_barrier(DMA_copy_lambda);
    }
      DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, 12, 20, 32, 20, 32, 0, 0, 0, 0, 0, 0, 8);
      DMA_copy_y.loc = (l1_buffer + 7688) + db_y;
      DMA_copy_y.number_of_2d_copies = y_tile_size_h;
      DMA_copy_y.number_of_1d_copies = y_tile_size_w;
      DMA_copy_y.length_1d_copy = y_length_nof_byte;
      dory_dma_memcpy_async(DMA_copy_y);   
    // update prev iterators
    db_state_y = ! db_state_y; 
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
    dory_cores_barrier();
  }

  // wait for final write
  dory_dma_barrier(DMA_copy_y);
  dory_dma_deallocate(dory_dma_channel);
}
