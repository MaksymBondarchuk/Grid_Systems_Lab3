#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "linalg.h"

/* Desired calculation precision */
const double epsilon = 0.001;

/* Compute the next approximation with Jacobi iteration */
void jacobi_iteration(
    int start_row,                     // First row index of the matrix part
    struct my_matrix *mat_A_part,      // Part of rows of coefficient matrix
    struct my_vector *b,               // b-vector
    struct my_vector *vec_prev_x,      // Previous approximation
    struct my_vector *vec_next_x_part, // Part of the next approximation vector
                                       // (returned by the function)
    double *residue_norm_part)         // Value of norm in the previous step
                                       // (returned by the function)
{
  /* Accumulator of the norm of the current part of calculation */
  double my_residue_norm_part = 0.0;

  /* Calculate the next approximation vector */
  for(int i = 0; i < vec_next_x_part->size; i++)
  {
    double sum = 0.0;
    for(int j = 0; j < mat_A_part->cols; j++)
    {
      if(i + start_row != j)
      {
        sum += mat_A_part->data[i * mat_A_part->cols + j] * vec_prev_x->data[j];
      }
    }

    sum = b->data[i + start_row] - sum;

    vec_next_x_part->data[i] =
      sum / mat_A_part->data[i * mat_A_part->cols + i + start_row];

    /* Calculate norm of the previous step */
    sum = -sum + mat_A_part->data[i * mat_A_part->cols + i + start_row] *
                 vec_prev_x->data[i + start_row];
    my_residue_norm_part += sum * sum;
  }

  *residue_norm_part = my_residue_norm_part;
}

/* Main function */
int main(int argc, char *argv[])
{
  const char *input_file_MA = "MA.txt";
  const char *input_file_b = "b.txt";
  const char *output_file_x = "x.txt";

  /* Initialize MPI */
  MPI_Init(&argc, &argv);

  /* Get the total number of tasks and the current task rank */
  int np, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Input data in task 0 */
  struct my_matrix *MA;
  struct my_vector *b;
  int N;
  if(rank == 0)
  {
    MA = read_matrix(input_file_MA);
    b = read_vector(input_file_b);

    if(MA->rows != MA->cols) {
      fatal_error("Matrix is not square!", 4);
    }
    if(MA->rows != b->size) {
      fatal_error("Dimensions of matrix and vector don't match!", 5);
    }
    N = b->size;
  }

  /* Send matrix and vector dimensions from the task 0 to all other tasks */
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  /* Reserve memory to store right hand side vector b */
  if(rank != 0)
  {
    b = vector_alloc(N, .0);
  }

  /* Calculate parts of vectors and matrix to be stored in every task.
   * Assume that N = k*np. Allocate memory to store parts
   * of vectors and matrix and initialize them. */
  int part = N / np;
  struct my_matrix *MAh = matrix_alloc(part, N, .0);
  struct my_vector *oldx = vector_alloc(N, .0);
  struct my_vector *newx = vector_alloc(part, .0);

  /* Split matrix 'MA' into parts by 'part' rows and send parts to all tasks.
   * Return memory of matrix 'MA'. */
  if(rank == 0)
  {
    MPI_Scatter(MA->data, N*part, MPI_DOUBLE,
                MAh->data, N*part, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    free(MA);
  }
  else
  {
    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL,
                MAh->data, N*part, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
  }
  /* Send vector b */
  MPI_Bcast(b->data, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* Calculate norm of vector b and send it to all tasks */
  double b_norm = 0.0;
  if(rank == 0)
  {
    for(int i = 0; i < b->size; i++)
    {
      b_norm = b->data[i] * b->data[i];
    }
    b_norm = sqrt(b_norm);
  }
  MPI_Bcast(&b_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* Value of iteration stopping criterion */
  double last_stop_criteria;
  /* Main cycle of Jacobi iteration */
  for(int i = 0; i < 1000; i++)
  {
    double residue_norm_part;
    double residue_norm;

    jacobi_iteration(rank * part, MAh, b, oldx, newx,
        &residue_norm_part);

    /* Calculate the total sum of the residual vector */
    MPI_Allreduce(&residue_norm_part, &residue_norm, 1,
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    residue_norm = sqrt(residue_norm);

    /* Check iteration stopping criterion. Since norm calculated
     * in the current step is for previous step approximation,
     * the result is the approximation obtained in the previous step. */
    last_stop_criteria = residue_norm / b_norm;
    if(last_stop_criteria < epsilon)
    {
      break;
    }

    /* Gather current approximation of vector of variables */
    MPI_Allgather(newx->data, part, MPI_DOUBLE,
        oldx->data, part, MPI_DOUBLE, MPI_COMM_WORLD);
  }

  /* Output the result */
  if(rank == 0)
  {
    write_vector(output_file_x, oldx);
  }

  /* Free allocated resources and finalize MPI environment */
  free(MAh);
  free(oldx);
  free(newx);
  free(b);
  return MPI_Finalize();
}

