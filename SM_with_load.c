#include <ida/ida.h> /* prototypes for IDA fcts., consts.    */
#include <math.h>
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <stdio.h>
#include <complex.h>
#include <sundials/sundials_math.h> /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of sunrealtype, sunindextype      */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to Newton SUNNonlinearSolver  */

/* Problem Constants */

#define NEQ  8

#define ZERO  SUN_RCONST(0.0)
#define ONE   SUN_RCONST(1.0)
#define SQRT3 SUN_RCONST(1.7320508075688772)

/* Macro to define dense matrix elements, indexed from 1. */

#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

/* Prototypes of functions called by IDA */

int respower(sunrealtype t, N_Vector y, N_Vector yp, N_Vector res, void* user_data);
//int jacrob(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp,
//           N_Vector resvec, SUNMatrix JJ, void* user_data, N_Vector tempv1,
//           N_Vector tempv2, N_Vector tempv3);

/* Prototypes of private functions */
static int check_retval(void* returnvalue, const char* funcname, int opt);

struct _user_data {
    sunrealtype w_base, V_base, T_base, P_load, Q_load, Ggnd, rstr,
                R_gen, X_gen, tag, E0, phiG, cosn, I_base_gen, pt, xmdm, dpu, addmt;
};
typedef struct _user_data* UserData;

/*
 *--------------------------------------------------------------------
 * Main Program
 *--------------------------------------------------------------------
 */

#define ICS_PARS_FILENAME "DAE_ICs_pars.txt"

#define S_BASE SUN_RCONST(1.0e6)
#define V_BASE SUN_RCONST(10.0e3)

int main(void)
{
  void* mem;
  N_Vector y, yp, avtol, res;
  sunrealtype rtol, *yval, *ypval, *atval;
  sunrealtype tstart, tstop, tout, tret, t, dt;
  int i, retval, flag;
  SUNMatrix A;
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
  SUNContext ctx;
  FILE *fid;
  UserData params;

  mem = NULL;
  y = yp = avtol = NULL;
  yval = ypval = atval = NULL;
  A                    = NULL;
  LS                   = NULL;
  NLS                  = NULL;

  /* Open the file containing initial conditions and parameters */
  fid = fopen(ICS_PARS_FILENAME, "r");
  if (fid == NULL) {
      fprintf(stderr, "%s: no such file.\n", ICS_PARS_FILENAME);
      exit(1);
  }

  params = (UserData) malloc(sizeof(struct _user_data));
  if (params == NULL) {
      fprintf(stderr, "Cannot allocate user data structure.\n");
      exit(1);
  }

  /* Create SUNDIALS context */
  retval = SUNContext_Create(SUN_COMM_NULL, &ctx);
  if (check_retval(&retval, "SUNContext_Create", 1)) { return (1); }

  /* Allocate N-vectors. */
  y = N_VNew_Serial(NEQ, ctx);
  if (check_retval((void*) y, "N_VNew_Serial", 0)) { return (1); }
  yp = N_VClone(y);
  if (check_retval((void*) yp, "N_VNew_Serial", 0)) { return (1); }
  avtol = N_VClone(y);
  if (check_retval((void*) avtol, "N_VNew_Serial", 0)) { return (1); }
  res = N_VClone(y);
  if (check_retval((void*) res, "N_VNew_Serial", 0)) { return (1); }

  /* Create and initialize  y, y', and absolute tolerance vectors. */
  yval = N_VGetArrayPointer(y);
  for (i=0; i<NEQ; i++)
      fscanf(fid, "%lg ", &yval[i]);

  ypval = N_VGetArrayPointer(yp);
  for (i=0; i<NEQ; i++)
      fscanf(fid, "%lg ", &ypval[i]);

  rtol = SUN_RCONST(1.0e-4);

  atval = N_VGetArrayPointer(avtol);
  for (i=0; i<NEQ; i++)
      atval[i] = SUN_RCONST(1.0e-4);

  /* Load the parameters values */
  fscanf(fid, "%lg\n", &params->w_base);
  fscanf(fid, "%lg\n", &params->V_base);
  fscanf(fid, "%lg\n", &params->T_base);
  fscanf(fid, "%lg\n", &params->P_load);
  fscanf(fid, "%lg\n", &params->Q_load);
  fscanf(fid, "%lg\n", &params->Ggnd);
  fscanf(fid, "%lg\n", &params->rstr);
  fscanf(fid, "%lg\n", &params->R_gen);
  fscanf(fid, "%lg\n", &params->X_gen);
  fscanf(fid, "%lg\n", &params->tag);
  fscanf(fid, "%lg\n", &params->E0);
  fscanf(fid, "%lg\n", &params->phiG);
  fscanf(fid, "%lg\n", &params->cosn);
  fscanf(fid, "%lg\n", &params->I_base_gen);
  fscanf(fid, "%lg\n", &params->pt);
  fscanf(fid, "%lg\n", &params->xmdm);
  fscanf(fid, "%lg\n", &params->dpu);
  fscanf(fid, "%lg\n", &params->addmt);

  fclose(fid);

  /* Integration limits */
  tstart = ZERO;
  tstop  = SUN_RCONST(360.);
  dt     = SUN_RCONST(5.0e-3);

  /* Call IDACreate and IDAInit to initialize IDA memory */
  mem = IDACreate(ctx);
  if (check_retval((void*)mem, "IDACreate", 0)) { return (1); }

  retval = IDASetUserData(mem, params);
  if (check_retval(&retval, "IDASetUserData", 1)) { return (1); }

  retval = IDAInit(mem, respower, tstart, y, yp);
  if (check_retval(&retval, "IDAInit", 1)) { return (1); }

  /* Call IDASVtolerances to set tolerances */
  retval = IDASVtolerances(mem, rtol, avtol);
  if (check_retval(&retval, "IDASVtolerances", 1)) { return (1); }

  /* Create dense SUNMatrix for use in linear solves */
  A = SUNDenseMatrix(NEQ, NEQ, ctx);
  if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (1); }

  /* Create dense SUNLinearSolver object */
  LS = SUNLinSol_Dense(y, A, ctx);
  if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (1); }

  /* Attach the matrix and linear solver */
  retval = IDASetLinearSolver(mem, LS, A);
  if (check_retval(&retval, "IDASetLinearSolver", 1)) { return (1); }

  /* Create Newton SUNNonlinearSolver object. IDA uses a
   * Newton SUNNonlinearSolver by default, so it is unnecessary
   * to create it and attach it. It is done in this example code
   * solely for demonstration purposes. */
  NLS = SUNNonlinSol_Newton(y, ctx);
  if (check_retval((void*)NLS, "SUNNonlinSol_Newton", 0)) { return (1); }

  /* Attach the nonlinear solver */
  retval = IDASetNonlinearSolver(mem, NLS);
  if (check_retval(&retval, "IDASetNonlinearSolver", 1)) { return (1); }

  /* In loop, call IDASolve, print results, and test for error. */
  t = tstart;
  flag = 0;
  while (t < tstop) {
    tout = t + dt;
    retval = IDASolve(mem, tout, &tret, y, yp, IDA_NORMAL);
    if (check_retval(&retval, "IDASolve", 1)) { return (1); }
    if (t > 60 && ! flag) {
        params->P_load *= 1.01;
        flag = 1;
    }
    if (retval == IDA_SUCCESS) {
        fprintf(stdout, "%lg", t);
        for (i=0; i<NEQ; i++)
            fprintf(stdout, " %lg", yval[i]);
        fprintf(stdout, "\n");
    }
    t = tret;
  }

  /* Free memory */
  IDAFree(&mem);
  SUNNonlinSolFree(NLS);
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(res);
  N_VDestroy(avtol);
  N_VDestroy(yp);
  N_VDestroy(y);
  SUNContext_Free(&ctx);
  free(params);

  return (retval);
}

/*
 *--------------------------------------------------------------------
 * Functions called by IDA
 *--------------------------------------------------------------------
 */

/*
 * Define the system residual function.
 */

int respower(sunrealtype t, N_Vector y, N_Vector yp, N_Vector res, void* user_data)
{
  sunrealtype *yval, *ypval, *rval;
  sunrealtype w, uBr, uBi, iLr, iLi, iGr, iGi;
  sunrealtype wp, phip;
  sunrealtype n, te, tm, utr, uti, itr, iti, psir, psii, den;
  UserData params;

  yval  = N_VGetArrayPointer(y);
  ypval = N_VGetArrayPointer(yp);
  rval  = N_VGetArrayPointer(res);

  params = (UserData) user_data;

  w    = yval[0];
  uBr  = yval[2];
  uBi  = yval[3];
  iLr  = yval[4];
  iLi  = yval[5];
  iGr  = yval[6];
  iGi  = yval[7];
  wp   = ypval[0];
  phip = ypval[1];

  den = SQRT3 * (uBr * uBr + uBi * uBi);

  // mechanical torque
  n  = w / params->w_base;
  tm = params->pt / n - (params->xmdm + params->dpu * n - params->addmt);

  // electrical torque
  n    = 1; // neglect rotor speed variations
  utr  = uBr / params->V_base;
  uti  = uBi / params->V_base;
  itr  = iGr / params->I_base_gen;
  iti  = iGi / params->I_base_gen;
  psir = (uti + params->rstr * iti) / n;
  psii = - (utr + params->rstr * itr) / n;
  te   = (iti * psir - itr * psii) / params->cosn;

  rval[0] = wp - (tm - te) / params->tag * params->w_base;
  rval[1] = phip;
  rval[2] = iGr - (iLr + params->Ggnd * uBr);
  rval[3] = iGi - (iLi + params->Ggnd * uBi);
  rval[4] = iLr - ( params->P_load * uBr + params->Q_load * uBi) / den;
  rval[5] = iLi - (-params->Q_load * uBr + params->P_load * uBi) / den;
  rval[6] = uBr + params->R_gen * iGr - params->X_gen * iGi - params->E0 * cos(params->phiG);
  rval[7] = uBi + params->X_gen * iGr + params->R_gen * iGi - params->E0 * sin(params->phiG);

  return (0);
}

/*
 *--------------------------------------------------------------------
 * Private functions
 *--------------------------------------------------------------------
 */

static int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;
  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }
  else if (opt == 1)
  {
    /* Check if retval < 0 */
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return (1);
    }
  }
  else if (opt == 2 && returnvalue == NULL)
  {
    /* Check if function returned NULL pointer - no memory allocated */
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}

