#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "commander.h"
#include "peer_localization.h"

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"

#include "log.h"
#include "param.h"
#include "math3d.h"

#include "uart1.h"
#include "debug.h"

#define DEBUG_MODULE "APP"


const float distance = 0.5;  // distance between UAV and target in m
const float angle = 0.0;   // angle between UAV and target
const float tau = 1/100.0; // update rate

bool start_main = false;
uint8_t peer_id = 5;
float max_velo_v = 1.0f;

struct vec target_pos;
float target_yaw;

float x_d = 0;
float y_d = 0;
float z_d = 0;
float phi_d = 0;

const uint32_t baudrate_esp32 = 115200;

// helper for calculating the angle between two vectors
float calcAngleBetweenVectors(struct vec vec1, struct vec vec2)
{
  struct vec vec1_norm = vnormalize(vec1);
  struct vec vec2_norm = vnormalize(vec2);
  float dot = vdot(vec1_norm, vec2_norm);
  return acosf(dot);
}

// calculate the vector of the heading, angle in radians
struct vec calcHeadingVec(float radius, float angle)
{
  float x = (float) 0.0f + radius * cosf(angle);
  float y = (float) 0.0f + radius * sinf(angle);
  return mkvec(x, y, 0.0f);
}

void appMain()
{

  static setpoint_t setpoint;
  
  struct vec p_D = vzero();  // current position
  struct vec p_D_prime = vzero(); // next position

  // struct vec e_D = vzero(); // heading vector of UAV
  struct vec e_H_delta = vzero(); // heading vector of target multiplied by safety distance
  
  // struct vec v_D = vzero(); // velocity of UAV
  
  float estYawRad;  // estimate of current yaw angle
  //float theta_prime_D; // angle between current UAV heading and desired heading
  //float theta_D; // angle between current UAV heading and last heading
  // float omega_prime_D; // attitude rate


  logVarId_t idStabilizerYaw = logGetVarId("stabilizer", "yaw");

  logVarId_t idXEstimate = logGetVarId("stateEstimate", "x");
  logVarId_t idYEstimate = logGetVarId("stateEstimate", "y");
  logVarId_t idZEstimate = logGetVarId("stateEstimate", "z");

  // init UART
  DEBUG_PRINT("[DEBUG] Init UART...\n");
  uart1Init(baudrate_esp32);
  DEBUG_PRINT("[DEBUG] done!\n");

  while(1) {
    vTaskDelay(M2T(10));

    uint8_t dummy = 0x00;
    uint8_t uart_buffer[16];

  //  if (start_main) {
     DEBUG_PRINT("[DEBUG] Start main...\n");
    
    DEBUG_PRINT("[DEBUG] Waiting for UART message...\n");
    while (dummy != 0xBC)
    {
      uart1Getchar((uint8_t*)&dummy);
    }
    DEBUG_PRINT("[DEBUG] Got package from !\n");
    
    uart1Getchar((uint8_t*)&dummy);
    uint8_t length = dummy;
    if (length == sizeof(uart_buffer))
    {
      for (uint8_t i = 0; i < length+1; i++)
      {
        uart1Getchar((uint8_t*)&uart_buffer[i]);
      }

      DEBUG_PRINT("[DEBUG] Read package from UART!:\n");
      for (uint8_t i = 0; i < length; i++)
      {
        DEBUG_PRINT("%02X", uart_buffer[i]);
      }
      DEBUG_PRINT("\n");
    }    

      // evtl. Schleife für UART Daten empfangen von Rest des Controller trennen
      // Daten nicht überschreiben, wenn unvollständig übermitteln

      p_D.x = logGetFloat(idXEstimate);
      p_D.y = logGetFloat(idYEstimate);
      p_D.z = logGetFloat(idZEstimate);

      int32_t x = *(int32_t *)(uart_buffer + 0);
      int32_t y = *(int32_t *)(uart_buffer + 4);
      int32_t z = *(int32_t *)(uart_buffer + 8);
      int32_t phi = *(int32_t *)(uart_buffer + 12);

      x_d = (float)x * 2.46902e-05f + 1.02329e+00f;
      y_d = (float)y * 2.46902e-05f + 7.05523e-04f;
      z_d = (float)z * 2.46902e-05f + 2.68245e-01f;
      phi_d = (float)phi * 2.46902e-05f + 5.60173e-04f;

      // DEBUG_PRINT("[DEBUG] Conversion worked?: %ld, %ld, %ld, %ld\n", x, y, z, phi);
      DEBUG_PRINT("[DEBUG] Received coordinates: %f, %f, %f, %f\n", (double)x_d, (double)y_d, (double)z_d, (double)phi_d);

      // DEBUG_PRINT("[DEBUG] Conversion to uint32 worked? %lu\n", x);  
      // velocity control
      // Query position of our target
      // if (peerLocalizationIsIDActive(peer_id))
      // {
      // peerLocalizationOtherPosition_t* target = peerLocalizationGetPositionByID(peer_id);

      // target_pos = mkvec(target->pos.x, target->pos.y, target->pos.z);
      // target_yaw = target->yaw;
      target_pos = mkvec(x_d, y_d, z_d);

      estYawRad = radians(logGetFloat(idStabilizerYaw));    // get the current yaw in degrees
      struct quat q = rpy2quat(mkvec(0,0,estYawRad));
      target_pos = vadd(p_D, qvrot(q, target_pos));

      struct vec target_drone_global = vsub(target_pos, p_D);
      target_yaw = atan2f(target_drone_global.y, target_drone_global.x);

      setpoint.mode.yaw = modeAbs;
      setpoint.attitude.yaw = degrees(target_yaw);
     
      // target_yaw = phi_d;

      // z is kept at same height as target
      setpoint.mode.z = modeAbs;
      setpoint.position.z = target_pos.z;

      // position is handled given position and attitude rate

      setpoint.mode.x = modeAbs;
      setpoint.mode.y = modeAbs;


      // eq 6
      e_H_delta = calcHeadingVec(1.0f*distance, target_yaw-M_PI_F);//angle); 
      // radius is 1 * distance
      // angle is set to the current yaw angle (rad) of the target

      p_D_prime = vadd(target_pos, e_H_delta);
      setpoint.position.x = 0.0f;//p_D_prime.x;
      setpoint.position.y = 0.0f;//p_D_prime.y;
      setpoint.position.z = 1.4f;//p_D_prime.z;

      // velocity control -> produces oscillation of the CF in x and y direction
      // // eq 7
      // struct vec v_H = vzero(); // target velocity, set to 0 since we don't have this information yet
      // v_D = vdiv(vsub(p_D_prime, p_D), tau);
      // v_D = vadd(v_D, v_H);
      // //v_D = vclamp(v_D, vneg(max_velocity), max_velocity); // -> debugging, doesn't preserve the direction of the vector
      // v_D = vclampnorm(v_D, max_velo_v);

      // setpoint.velocity.x = v_D.x;
      // setpoint.velocity.y = v_D.y;


      // heading control
      // eq 8
      // e_D = calcHeadingVec(1.0f, estYawRad);           // radius is 1, angle is current yaw 

      // struct vec target_vector = vsub(p_D, target_pos);
      // float angle_target = atan2f(target_vector.y, target_vector.x);

      //theta_prime_D = calcAngleBetweenVectors(e_D, vsub(target_pos, p_D));
      
      //theta_D = calcAngleBetweenVectors(e_D, vsub(target_pos, p_D_prime));  //estYawRad;
      
      // eq 9
      // omega_prime_D = (theta_prime_D - theta_D) / tau;     // <-- incorrect, the difference does not always provide the shortest angle between the two thetas
      // omega_prime_D = shortest_signed_angle_radians(estYawRad, angle_target) / tau;
      // omega_prime_D = clamp(omega_prime_D, -0.8f, 0.8f);

      // setpoint.mode.yaw = modeVelocity;
      // setpoint.attitudeRate.yaw = degrees(omega_prime_D);


      setpoint.velocity_body = false;  // world frame

      // send a new setpoint to the UAV
      // DEBUG_PRINT("[DEBUG] Setpoint:\n");
      // DEBUG_PRINT("[DEBUG] x: %f, y: %f, z: %f, ");
      //       x_limit: [-1.0, 1.0] # m
      // y_limit: [-1.0, 1.0] # m
      // z_limit: [0.0, 2.5]
      setpoint.position.x = clamp(setpoint.position.x, -0.8f, 0.8f);
      setpoint.position.y = clamp(setpoint.position.y, -0.8f, 0.8f);
      setpoint.position.z = clamp(setpoint.position.z, 0.0f, 2.0f);

      if (start_main) {
      commanderSetSetpoint(&setpoint, 3);

      // update current position
      //p_D = p_D_prime;
      //}//end debug if

      }//end if
  }//end while
}//end main


/**
 * Parameters to set the start flag, peer id and max velocity
 * for the frontnet-like controller.
 */
PARAM_GROUP_START(frontnet)
/**
 * @brief Estimator type Any(0), complementary(1), kalman(2) (Default: 0)
 */
PARAM_ADD_CORE(PARAM_UINT8, start, &start_main)
PARAM_ADD_CORE(PARAM_UINT8, cfid, &peer_id)
PARAM_ADD_CORE(PARAM_FLOAT, maxvelo, &max_velo_v)


PARAM_GROUP_STOP(frontnet)

// add new log group for local variables
LOG_GROUP_START(frontnet)
LOG_ADD(LOG_FLOAT, targetx, &target_pos.x)
LOG_ADD(LOG_FLOAT, targety, &target_pos.y)
LOG_ADD(LOG_FLOAT, targetz, &target_pos.z)
LOG_ADD(LOG_FLOAT, targetyaw, &target_yaw)

LOG_ADD(LOG_FLOAT, x_uart, &x_d)
LOG_ADD(LOG_FLOAT, y_uart, &y_d)
LOG_ADD(LOG_FLOAT, z_uart, &z_d)
LOG_ADD(LOG_FLOAT, phi_uart, &phi_d)
LOG_GROUP_STOP(frontnet)
