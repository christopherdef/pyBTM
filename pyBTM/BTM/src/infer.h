#ifndef _INFERLDA_H
#define _INFERLDA_H

#include <string>

#include "pvec.h"
#include "pmat.h"

#include "doc.h"

using namespace std;

class Infer {
private:
  int K;
  int n_iter;			// maximum number of iteration of Gibbs Sampling
  double alpha;			// hyperparameters of p(z)
  double beta;			// hyperparameters of p(w|z)

  string type;			// infer type
  
  string dfile;			// inference docs
  Pvec<double> pz;	    // p(z) = theta
  Pmat<double> pw_z;   // p(w|z) = phi, size K * M

private:
  void load_para(string model_dir);
  
  void doc_infer(const Doc& doc, Pvec<double>& pz_d);  
  void doc_infer_sum_b(const Doc& doc, Pvec<double>& pz_d);
  void doc_infer_sum_w(const Doc& doc, Pvec<double>& pz_d);
  void doc_infer_mix(const Doc& doc, Pvec<double>& pz_d);
  
  // compute condition distribution p(z|w, d) with p(w|z) fixed
  void compute_pz_dw(int w, const Pvec<double>& pz_d, Pvec<double>& p); 
  
public:
  Infer(string type, int K, double a, double b, int n_iter): 
    type(type), K(K), alpha(a), beta(b), n_iter(n_iter) {}

  void run(string docs_pt, string model_dir);
};

#endif
