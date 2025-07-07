After having MRS UAV System running, the controller can be run by following these steps:

1 - Go to the folder where the code is.
2 - Run devel/setup.bash
3 - Run roslaunch multi_drone uav_sequence.launch

The code supports currently the use of three drones.

(To run the MRS UAV System after installed, do:
1 - roscd mrs_uav_gazebo_simulation/tmux
2 - cd three_drones/
3 - ./start.sh)

This controller is shown in the conference paper:
"Vision-Based Target Pursuit and Formation Maintenance for UAV Teams using Nonlinear Model Predictive Control", João Matias, Rodrigo Ventura, Meysam Basiri

This controller runs on MRS UAV System:
Baca, T., Petrlik, M., Vrba, M., Spurny, V., Penicka, R., Hert, D., & Saska, M. (2020, August 18). The MRS UAV System: Pushing the frontiers of reproducible research, real‑world deployment, and education with autonomous unmanned aerial vehicles. arXiv. https://arxiv.org/abs/2008.08050

# NMPC_Multi_UAV_Team_Target_Pursuit 
