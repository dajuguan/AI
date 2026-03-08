# Prerequisites
- Install Python Pytorch
- export the libpytorch lib in enviroment
```
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH}"
```

# How to run
```
cargo run 
```