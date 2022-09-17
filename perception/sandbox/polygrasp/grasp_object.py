#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""
import hydra
from storm_grasp_wrapper import storm_grasp
    
@hydra.main(config_path="conf", config_name="run_grasp")
def main(cfg):
    grasp_class = storm_grasp(cfg)
    if(grasp_class.init_fail is not False):
        print("init failed")
    # else:
        # grasp_class.find_plate()
    print(grasp_class.grasp_candidates(cfg))

if __name__ == "__main__":
    main()