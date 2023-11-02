/*
 * test_template.c
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
#include "mem.h"
#include "network.h"
#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/camera.h"
#include "bsp/camera/himax.h"
#include "bsp/buffer.h"


// #define VERBOSE 1
#define IMG_ORIENTATION 0x0101
#define CAMERA_WIDTH 324
#define CAMERA_HEIGHT 324
#define BUFF_SIZE (CAMERA_WIDTH*CAMERA_HEIGHT)
#define INPUT_HEIGHT 320 
#define INPUT_WIDTH 320 
// //streaming
// #define JPEG_STREAMER 1
// #define STREAM_WIDTH WIDTH
// #define STREAM_HEIGHT HEIGHT
static unsigned char *l2_camera_buff;
static PI_L2 uint8_t length = sizeof(int32_t)*2;
static struct pi_device camera;
// static struct pi_device uart;

static PI_L2 uint8_t magic = 0xBC;
uint8_t * l2_out_buffer; 
uint8_t * raw_out;
int32_t pix_center[2];
PI_L2 char* L2_output;

static void set_register(uint32_t reg_addr, uint8_t value)
{
  uint8_t set_value = value;
  pi_camera_reg_set(&camera, reg_addr, &set_value);
}


static int open_pi_camera_himax(struct pi_device *device)
{
  struct pi_himax_conf cam_conf;

  pi_himax_conf_init(&cam_conf);

  cam_conf.format = PI_CAMERA_QVGA;

  pi_open_from_conf(device, &cam_conf);
  if (pi_camera_open(device))
    return -1;

  // Rotate camera orientation
  set_register(IMG_ORIENTATION, 3);
  set_register(0x0104, 0x1);

  // uint8_t aeg = 0x00;
  // uint8_t aGain = 8;
  // uint8_t dGain = 1;
  // uint16_t exposure = 100;

  //  set_register(0x2100, aeg);  // AE_CTRL

  //     switch(aGain) {
  //       case 8:
  //         set_register(0x0205, 0x30);
  //         break;
  //       case 4:
  //         set_register(0x0205, 0x20);
  //         break;
  //       case 2:
  //         set_register(0x0205, 0x10);
  //         break;
  //       case 1:
  //       default:
  //         set_register(0x0205, 0x00);
  //         break;
  //     }

  //     set_register(0x020E, (dGain >> 6)); // 2.6 int part
  //     set_register(0x020F, dGain & 0x3F); // 2.6 float part

      
  //     if (exposure < 2) {
  //       exposure = 2;
  //     }
  //     if (exposure > 0x0216 - 2) {
  //       exposure = 0x0216 - 2;
  //     }
  //     set_register(0x0202, (exposure >> 8) & 0xFF);    // INTEGRATION_H
  //     set_register(0x0203, exposure & 0xFF);    // INTEGRATION_L

  //     // This is needed for the camera to actually update its registers.
  //     set_register(0x0104, 0x1);
  return 0;
}


void cropImage(unsigned char *image_raw, unsigned char *image_cropped)
{
  for (uint16_t i = 0; i < 320; i++) // rows
    {
        for (uint16_t j = 0; j < 320; j++) //cols
        {    
            *(image_cropped+i*INPUT_WIDTH+j) = *(image_raw+(i+2)*CAMERA_WIDTH+j+2);
            // printf("oCB= %d, oIB = %d\n",(i*INPUT_WIDTH+j),((i+2)*CAMERA_WIDTH+j+2));
        }
    }
  // printf("Raw=%ld, Cropped=%ld",image_raw,image_cropped) ;
}

void application(void * arg) {
/*
    Opening of Filesystem and Ram
*/
  mem_init();
  network_initialize();
  /*
    Allocating space for input
  */
  void *l2_buffer = pi_l2_malloc(215000);

  if (NULL == l2_buffer) {
#ifdef VERBOSE
    printf("ERROR: L2 buffer allocation failed.");
#endif
    pmsis_exit(-1);
  }
#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\tOk\n", (unsigned int)l2_buffer);
#endif

/*
    Allocating space for camera buffer
*/
  L2_output = pi_l2_malloc((int32_t) 4);
  uint32_t camera_buffer_size = BUFF_SIZE * sizeof(unsigned char);
  l2_camera_buff = (unsigned char *)pi_l2_malloc(camera_buffer_size);
  if (l2_camera_buff == NULL){printf("\nCamera Buffer Failed");return -1;}
  else{
  // printf("\nCamera Buffer alloc initial\t@ 0x%08x:\tOk\n", (unsigned int)l2_camera_buff);
  }

// Allocating space for the raw output
	raw_out = (uint8_t *) pi_l2_malloc(3200*sizeof(uint8_t));
	if (raw_out==0) {
		// printf("Failed to allocate memory for raw_out (%ld bytes)\n", 3200*sizeof(uint8_t));
		return -1;
	}
  else{
  // printf("\nRaw Output Buffer alloc (%ld bytes) initial\t@ 0x%08x:\tOk\n", 3200*sizeof(uint8_t),(unsigned int)raw_out);
  }
  l2_out_buffer = raw_out;

// Allocate the output tensor
	// pix_center = (int32_t *) pi_l2_malloc(2*sizeof(int32_t));
	// if (pix_center==0) {
	// 	printf("Failed to allocate memory for pix_center (%ld bytes)\n", 2*sizeof(uint32_t));
	// 	return -1;
	// }
  // else{
  // // printf("\nOutput Buffer alloc initial\t@ 0x%08x:\tOk\n", (unsigned int)pix_center);
  // }
  

// Open the Himax camera
if (open_pi_camera_himax(&camera))
{
    //printf("Failed to open camera\n");
    pmsis_exit(-1);
}

// UART configuration for debug printfs
struct pi_uart_conf uart_conf;
struct pi_device uart_device;
pi_uart_conf_init(&uart_conf);
uart_conf.baudrate_bps = 115200;
pi_open_from_conf(&uart_device, &uart_conf);
if (pi_uart_open(&uart_device))
{
  //printf("[UART] open failed !\n");
  pmsis_exit(-1);
}
pi_uart_write(&uart_device, &magic, 1);
// printf("\n[UART] Sent %c!\n",magic);
size_t l2_input_size = 102400;
size_t input_size = 1000000;

void *ram_input = ram_malloc(input_size);
// unsigned long long start = pi_time_get_us();
// load_file_to_ram(ram_input,"inputs.hex");
// ram_read(l2_buffer, ram_input, l2_input_size);
// network_run(l2_buffer, 215000, l2_out_buffer, 0);

// while(1)
// { 
//   // printf("In Loop\n") ;
//   unsigned long long start = pi_time_get_us();
//   // Open Camera , Capture Image, Close camera
//   pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
//   pi_camera_capture(&camera, l2_camera_buff, BUFF_SIZE);
//   pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
//   // printf("Camera Opened\n") ;
//   // Remove the dead pixels and crop image 324*324 to 320*320
//   cropImage(l2_camera_buff, l2_camera_buff);
//   network_run(l2_camera_buff, 215000, l2_out_buffer, 0);

//   //printf("Network Output: ");
  uint8_t max=0;
  uint32_t max_ind;
  uint8_t z = 0;
//   //Find the maximum value from 40*40*2 grid, stored as (ch1,ch2,ch1,ch2....)

  for(int i = 0; i < 3200; i++)
  { 
    if(l2_out_buffer[i]!=0)
    {
    // printf("%d:%d\n", i,l2_out_buffer[i]);
      if(l2_out_buffer[i]>max)
      {
        max = l2_out_buffer[i];
        max_ind = i;
      }
    }  
  }
  z = l2_out_buffer[max_ind+1];
  max_ind = max_ind/2;

  // Calculate the center pixels from the predictions
  pix_center[1] = 22;
  // ((max_ind/40)*8); // curH corresponds to y, row
  pix_center[0] = 33;
  // ((max_ind%40)*8); // curW corresponds to x, col
  
  unsigned long long end = pi_time_get_us();
  pi_uart_write(&uart_device, &magic, 1);
  pi_uart_write(&uart_device, &length, 1);
  pi_uart_write(&uart_device, (char*)pix_center, 2*sizeof(uint32_t));

  // printf("\nPix co-ordinates are (%d,%d), measured time: %lld, dist is %d\n",pix_center[0],pix_center[1],end-start,z);
// }
//pi_uart_write(&uart_device, &pix_center, 8);


// ram_free(ram_input, input_size);
// network_terminate();
// pi_l2_free(l2_buffer, 215000);
// pi_l2_free(l2_camera_buff, camera_buffer_size);
}

int main () {
#ifndef TARGET_CHIP_FAMILY_GAP9
  PMU_set_voltage(1000, 0);
#endif
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
  pi_time_wait_us(10000);


  pmsis_kickoff((void*)application);
  return 0;
}
