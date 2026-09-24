Setup:
  # Create virtual env
  python -m venv .venv
  # activate virtual environment
  source ./bin/activate
  # install dependencies
  pip install
  ./run + args


Usage:
    "low_pass_path", help="relative path to the image in the dataset"
    "high_pass_path", help="relative path to the image in the dataset"
    "-a", "--alignment_algorithm", help="harris_align = 0 | middle_out_align = 1", default=0 
    "-lk", "--low_kernel_size", help="low pass filter kernel size", default=9
    "-ls", "--low_pass_sigma", help="low pass filter sigma", default=3
    "-hk", "--high_kernel_size", help="high pass filter kernel size", default=5
    "-hs", "--high_pass_sigma", help="low pass filter sigma", default=1.4
    "-m", "--merge_alpha", help="strength of high-pass image in percentages 0-1", default=0.5

Output:
  Currently I have it set up so that matplotlib prints on my terminal. But since I'm pretty sure
  it only works on Kitty terminal emulator, I made it so that it prints in the working directory as
  ./output_img.png

