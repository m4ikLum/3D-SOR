/****************************************************************************
   3D SOR Implementation (Serial Version)

   gcc -O1 -std=gnu11 serial_SOR3D.c -lm -o serial_SOR3D
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

#define GHOST 2   /* 2 extra rows/columns/slices for "ghost zone". */

#define A   8    /* coefficient of x^2 */
#define B   8    /* coefficient of x */
#define C   8    /* constant term */

#define NUM_TESTS 5

#define MINVAL   0.0
#define MAXVAL  10.0

#define TOL 0.00001
#define OMEGA 1.83

typedef double data_t;

typedef struct {
  long int n;      //size of cube (n x n x n)
  data_t *data;    //3D data stored as 1D array
} cube_rec, *cube_ptr;

/* Prototypes */
cube_ptr new_cube(long int n);
int init_cube_rand(cube_ptr c);
data_t *get_cube_start(cube_ptr c);
double fRand(double fMin, double fMax);
void SOR3D_serial(cube_ptr c, int *iterations);

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/*****************************************************************************/
int main(int argc, char *argv[])
{
  struct timespec time_start, time_stop;
  double time_stamp[NUM_TESTS];
  int convergence[NUM_TESTS];
  int *iterations;

  long int x, n;
  
  printf("3D SOR Serial Implementation\n");
  printf("OMEGA = %0.2f\n", OMEGA);

  //allocate space for return value 
  iterations = (int *) malloc(sizeof(int));

  printf("Testing 3D SOR serial implementation\n");
  for (x = 0; x < NUM_TESTS; x++) {
    n = A*x*x + B*x + C;
    printf("  iter %ld cube size = %ld x %ld x %ld\n", x, n, n, n);
    
    //create and initialize cube
    cube_ptr cube = new_cube(n + GHOST);
    init_cube_rand(cube);
    
    //run SOR and time it 
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR3D_serial(cube, iterations);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    
    time_stamp[x] = interval(time_start, time_stop);
    convergence[x] = *iterations;
    
    //clean up 
    free(cube->data);
    free(cube);
  }

  printf("\nResults:\n");
  printf("size (n x n x n), time (cycles), iterations\n");
  for (x = 0; x < NUM_TESTS; x++) {
    n = A*x*x + B*x + C;
    printf("%4ld x %4ld x %4ld, %10.4g, %4d\n", 
           n, n, n, 
           (double)CPNS * 1.0e9 * time_stamp[x],
           convergence[x]);
  }

  //clean up
  free(iterations);

  return 0;
}

/*********************************/

//create 3D cube of specified size per dimension 
cube_ptr new_cube(long int n)
{
  //allocate and declare header structure 
  cube_ptr cube = (cube_ptr) malloc(sizeof(cube_rec));
  if (!cube) {
    return NULL;  /* Couldn't allocate storage */
  }
  cube->n = n;

  //allocate and declare array 
  if (n > 0) {
    data_t *data = (data_t *) calloc(n*n*n, sizeof(data_t));
    if (!data) {
      free((void *) cube);
      printf("\n COULDN'T ALLOCATE STORAGE \n");
      return NULL;  //couldn't allocate storage
    }
    cube->data = data;
  }
  else cube->data = NULL;

  return cube;
}

//initialize cube with random data 
int init_cube_rand(cube_ptr c)
{
  long int i, n, size;
  data_t *data;

  n = c->n;
  data = c->data;
  size = n * n * n;
  
  //set random nunbers 
  srandom(n);
  
  for (i = 0; i < size; i++) {
    data[i] = (data_t)(fRand((double)(MINVAL), (double)(MAXVAL)));
  }
  
  return 1;
}

data_t *get_cube_start(cube_ptr c)
{
  return c->data;
}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

//index calculation for 3D cube 
int idx3D(int i, int j, int k, int n) 
{
  //i is x, j is y, k is z 
  return (k * n * n) + (j * n) + i;
}

/************************************/

void SOR3D_serial(cube_ptr c, int *iterations)
{
  int i, j, k;
  int n = c->n;
  data_t *data = c->data;
  double change, total_change = 1.0e10;   /* start w/ something big */
  int iters = 0;
  int idx;

  //main iteration loop
  while ((total_change / (double)(n*n*n)) > (double)TOL) {
    iters++;
    total_change = 0.0;
    
    //process all interior 6 points of the cube
    for (k = 1; k < n-1; k++) {        //z-coordinate 
      for (j = 1; j < n-1; j++) {      //y-coordinate 
        for (i = 1; i < n-1; i++) {    //x-coordinate 
          idx = idx3D(i, j, k, n);
          
          //calculate change based on 6 neighbors
          change = data[idx] - (1.0/6.0) * (
            data[idx3D(i-1, j, k, n)] +  //left neighbor (x-1) 
            data[idx3D(i+1, j, k, n)] +  //right neighbor (x+1) 
            data[idx3D(i, j-1, k, n)] +  //down neighbor (y-1) 
            data[idx3D(i, j+1, k, n)] +  //up neighbor (y+1) 
            data[idx3D(i, j, k-1, n)] +  //back neighbor (z-1) 
            data[idx3D(i, j, k+1, n)]    //front neighbor (z+1) 
          );
          
          //apply SOR update 
          data[idx] -= change * OMEGA;
          
          //track convergence 
          if (change < 0) {
            change = -change;
          }
          total_change += change;
        }
      }
    }
    
    //check for divergence 
    int last_idx = idx3D(n-2, n-2, n-2, n);
    if (fabs(data[last_idx]) > 10.0 * (MAXVAL - MINVAL)) {
      printf("SOR3D: SUSPECT DIVERGENCE iter = %d\n", iters);
      break;
    }
  }
  
  *iterations = iters;
  printf("    SOR3D_serial() done after %d iters\n", iters);
}
