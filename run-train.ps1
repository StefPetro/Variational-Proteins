.\venv\Scripts\activate

param ([int] $process)    # -process # 
param([switch] $shutdown) # -shutdown

## N_VAE
if ( 1 -eq $process ) 
{
    write-host "train.py"
    python N_VAE/train.py
} 
elseif ( 2 -eq $process )
{
    write-host "train_basic.py"
    python N_VAE/train_basic.py
}
## S_VAE
elseif ( 3 -eq $process )
{
    write-host "train_s_vae.py"
    python S_VAE/train_s_vae.py
}
elseif ( 4 -eq $process )
{   
    write-host "train_s_vae.py"
    python S_VAE/train_s_vae_basic.py
}
else 
{
    write-host "You need to specify which process to run -process #"
}

# Shutdown PC after execution 
if ( $shutdown )
{
    Stop-Computer
}
