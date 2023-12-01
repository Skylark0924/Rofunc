import rofunc as rf

rofunc_path = rf.oslab.get_rofunc_path()
if not os.path.exists(os.path.join(rofunc_path, "simulator/assets/urdf/ycb/")):


args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

object_name = "Cabinet"
rf.object.show(args, object_name)
