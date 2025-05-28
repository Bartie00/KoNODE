This code is implemented and improved based on the `torchdiffeq` library.

We use the NON-LINEAR OSCILLATOR data example, with the data stored in `train_data.npy`, `test_data.npy`, and `val_data.npy`.

To set up the environment, please refer to `setup_env.sh`. 

Run the code according to the following example:

```sh
python main.py --gpu 0 --w_size 10 --solver euler 
```
