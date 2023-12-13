# opencv_ext
Extension of opencv that more closely mimics other interfaces for ergonomics and portability

Functions are broken up into namespace categories, and everything exists under the cvx namespace, and GPU functionality is similar but with 'cuda' between cvx and the appropriate namespace.
- core (under the root namespace)
- matlab
- common
- cuda
  - core (under the cuda namespace)
  - matlab
  - common


NOTE: The API is not considered stable, as namespaces (and possibly function signatures) might change over time.
