/****************************************************************************
   3D SOR Implementation (Parallel Version - Y-axis Partitioning)

   gcc -O1 -std=gnu11 3DSORy.c -lm -pthread -o parallel_SOR3D_y
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define CPNS 3.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GhZ GPU, this would be 3.2 */

#define GHOST 2   /* 2 extra rows/columns/slices for "ghost zone". */

#define A   2    /* coefficient of x^2 */
#define B   4    /* coefficient of x */
#define C   0    /* constant term */

#define NUM_TESTS 9

#define MINVAL   0.0
#define MAXVAL  10.0

#define TOL 0.00001
#define OMEGA 1.83

//Default number of threads
#define DEFAULT_NUM_THREADS 4

typedef double data_t;

typedef struct {
  long int n;      //size of cube (n x n x n)
  data_t *data;    //3D data stored as 1D array
} cube_rec, *cube_ptr;

//thread arguments structure 
typedef struct {
  cube_ptr cube;
  int thread_id;
  int num_threads;
  double *local_changes;
  pthread_barrier_t *barrier;
  int *continue_iteration;
  int *total_iterations;
} thread_args;

/* Prototypes */
cube_ptr new_cube(long int n);
int init_cube_rand(cube_ptr c);
data_t *get_cube_start(cube_ptr c);
double fRand(double fMin, double fMax);
void SOR3D_parallel(cube_ptr c, int *iterations, int num_threads);
void *SOR3D_thread_work(void *args);

//synchronization objects 
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

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
  int num_threads = DEFAULT_NUM_THREADS;

  long int x, n;
  
  //check for command line argument for number of threads 
  if (argc > 1) {
    num_threads = atoi(argv[1]);
    if (num_threads <= 0) {
      printf("Invalid number of threads. Using default: %d\n", DEFAULT_NUM_THREADS);
      num_threads = DEFAULT_NUM_THREADS;
    }
  }
  
  printf("3D SOR Parallel Implementation (Y-axis Partitioning)\n");
  printf("OMEGA = %0.2f, Number of Threads = %d\n", OMEGA, num_threads);

  //allocate space for return value 
  iterations = (int *) malloc(sizeof(int));

  printf("Testing 3D SOR parallel implementation\n");
  for (x = 0; x < NUM_TESTS; x++) {
    n = A*x*x + B*x + C;
    printf("  iter %ld cube size = %ld x %ld x %ld\n", x, n, n, n);
    
    //create and initialize cube 
    cube_ptr cube = new_cube(n + GHOST);
    init_cube_rand(cube);
    
    //run SOR and time it 
    clock_gettime(CLOCK_REALTIME, &time_start);
    SOR3D_parallel(cube, iterations, num_threads);
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

  //clean up (here we go again)
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
    return NULL;  //couldn't allocate storage :(
  }
  cube->n = n;

  //alocate and declare array 
  if (n > 0) {
    data_t *data = (data_t *) calloc(n*n*n, sizeof(data_t));
    if (!data) {
      free((void *) cube);
      printf("\n COULDN'T ALLOCATE STORAGE \n");
      return NULL;  //couldn't allocate storage (AGAIN LOL)
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
  
  //set random shits for reproducibility
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

//3D SOR - Parallel implementation with Y-axis partitioning 
void SOR3D_parallel(cube_ptr c, int *iterations, int num_threads)
{
  pthread_t *threads;
  thread_args *args;
  pthread_barrier_t barrier;
  double *local_changes;
  int continue_iteration = 1;
  int total_iterations = 0;
  int t;
  
  //initialize barrier 
  pthread_barrier_init(&barrier, NULL, num_threads);
  
  //allocate thread handles and arguments
  threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
  args = (thread_args *) malloc(num_threads * sizeof(thread_args));
  local_changes = (double *) calloc(num_threads, sizeof(double));
  
  //create threads 
  for (t = 0; t < num_threads; t++) {
    args[t].cube = c;
    args[t].thread_id = t;
    args[t].num_threads = num_threads;
    args[t].local_changes = &local_changes[t];
    args[t].barrier = &barrier;
    args[t].continue_iteration = &continue_iteration;
    args[t].total_iterations = &total_iterations;
    
    pthread_create(&threads[t], NULL, SOR3D_thread_work, (void *)&args[t]);
  }
  
  //wait for all threads to complete 
  for (t = 0; t < num_threads; t++) {
    pthread_join(threads[t], NULL);
  }
  
  //clean up 
  pthread_barrier_destroy(&barrier);
  free(threads);
  free(args);
  free(local_changes);
  
  //set the output iterations value 
  *iterations = total_iterations;
  printf("    SOR3D_parallel() done after %d iters\n", total_iterations);
}

//thread worker function for SOR computation - Y-axis partitioning
void *SOR3D_thread_work(void *arg)
{
  thread_args *my_args = (thread_args *)arg;
  cube_ptr c = my_args->cube;
  int thread_id = my_args->thread_id;
  int num_threads = my_args->num_threads;
  double *my_change = my_args->local_changes;
  pthread_barrier_t *barrier = my_args->barrier;
  int *continue_iteration = my_args->continue_iteration;
  int *total_iterations = my_args->total_iterations;
  
  int i, j, k;
  int n = c->n;
  data_t *data = c->data;
  double change;
  double total_change;
  int idx;
  
  //calculate work division for j-slices (y-coordinate)
  int j_points = n - 2;  //exclude ghost zone
  int j_per_thread = j_points / num_threads;
  int j_remainder = j_points % num_threads;
  
  //calculate range of j values 
  int j_start = 1 + thread_id * j_per_thread;
  int j_end = j_start + j_per_thread;
  
  //last thread takes any remainder 
  if (thread_id == num_threads - 1) {
    j_end += j_remainder;
  }
  
  //main iteration loop 
  while (*continue_iteration) {
    //Reset local change counter
    *my_change = 0.0;
    
    //process assigned portion of the cube 
    for (k = 1; k < n-1; k++) {        //z-coordinate 
      for (j = j_start; j < j_end; j++) { //y-coordinate (partitioned)
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
          *my_change += change;
        }
      }
    }
    
    //wait for all threads to finish this iteration 
    pthread_barrier_wait(barrier);
    
    //thread 0 checks for convergence and updates global variables 
    if (thread_id == 0) {
      (*total_iterations)++;
      
      //sum up changes from all threads 
      total_change = 0.0;
      for (i = 0; i < num_threads; i++) {
        total_change += my_args[i].local_changes[0];
      }
      
      //check for convergence 
      if ((total_change / (double)(n*n*n)) <= (double)TOL) {
        *continue_iteration = 0;
      }
      
      //check for divergence (only needed in one thread) 
      int last_idx = idx3D(n-2, n-2, n-2, n);
      if (fabs(data[last_idx]) > 10.0 * (MAXVAL - MINVAL)) {
        printf("SOR3D: SUSPECT DIVERGENCE iter = %d\n", *total_iterations);
        *continue_iteration = 0;
      }
    }
    
    //wait for thread 0 to determine if we continue
    pthread_barrier_wait(barrier);
  }
  
  return NULL;
}
