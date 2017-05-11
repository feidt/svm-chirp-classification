import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import numpy as np
import os


dev = pycuda.autoinit.device
max_block_dim = dev.max_block_dim_x
print("max block size = " + str(max_block_dim))

blocks = 8 #8
block_size = 512 #max_block_dim
nb_values = blocks * block_size
print("number of sample points = " + str(nb_values))

start_time = drv.Event()
end_time = drv.Event()


# C function executed on the GPU
mod = SourceModule("""
  __device__ float pes_kernel_real(float w, float omega, float energy, float inter_energy, float delay, float width,float lifetime, float chirp)
  {
    float hbar = 0.6582;
    return (cos((w+chirp*w*w)*delay/hbar)*(w-inter_energy) - sin((w+chirp*w*w)*delay/hbar)*lifetime) *  1./pow(cosh(width*3.145*0.5 * (w-energy-omega)/hbar),1) * 1./( pow((w-inter_energy)/hbar,2) + lifetime*lifetime);
  }


  __global__ void calculate_ac(float *dest,float *delay, int nb_samples, float integral_range, float omega,
                               float width, float intermediate_energy, float inverse_lifetime, float chirp)
  {
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  float w_integral = 0;
  float dw = 2.*integral_range/((float) nb_samples);
  float energy = 0.;

  for(int n = 0; n < nb_samples; n++)
  {
    w_integral +=  pes_kernel_real(-integral_range + dw*n, omega, energy, intermediate_energy, delay[i], width,inverse_lifetime, chirp)+
                         pes_kernel_real(-integral_range + dw*n, omega, energy, intermediate_energy, 0, width,inverse_lifetime, chirp);
  }
  dest[i] = w_integral;
  }
""")

gpu_ac = mod.get_function("calculate_ac")


nb_samples = 500

max_delay_time = 200.
omega = 1.55 # angular frequency in eV ~ 800 nm laser pulse
width = 15. # temporal width in fs
integral_range = 3. #total integration range [-integral_range, integral_range] in eV

# create a delay time list
delay = np.linspace(-max_delay_time,max_delay_time,nb_values,dtype=np.float32)
print("delta_delay = " + str(max_delay_time*2./float(nb_values)))

# create an array for the calculated autocorrelation
dest = np.zeros(nb_values,dtype=np.float32)

labels = np.zeros(nb_samples,dtype=int)


path = str(os.getcwd() + "/data")
if not os.path.exists(path):
    os.makedirs(path)


# start timing
start_time.record()


# generate autocorrelation data for classification
for i in range(nb_samples):

    mu = (0.5-np.random.uniform(0.2,1))*1e-2
    chirp = np.random.randn()*1e-3 + mu

    intermediate_energy = 1.56
    inverse_lifetime = 0.04

    if chirp > 0:
        labels[i] = 1
    else:
        labels[i] = 0
    #print("chirp = " + str(chirp))
    #print("label = " + str(labels[i]))

    # perform the GPU calculation
    gpu_ac(drv.Out(dest),drv.In(delay), np.int32(nb_values), np.float32(integral_range), np.float32(omega), np.float32(width), np.float32(intermediate_energy),np.float32(inverse_lifetime),np.float32(chirp),grid=(blocks,1),block=(block_size,1,1))
    filename = os.getcwd() + "/data/" + str(i) + ".txt"
    np.savetxt(filename,np.c_[delay,dest])

# save target labels
np.savetxt(os.getcwd() + "/data/labels.txt", labels)
 # end timing
end_time.record()

# calculate computing time
end_time.synchronize()
secs = start_time.time_till(end_time)*1e-3
print("{} = {:.4f} {}".format("computing time", secs,"sec"))

#plt.figure()
#plt.plot(dest)
#plt.xlim(0,nb_values)
#plt.show()
