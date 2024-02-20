import bpy, bmesh
import math,os,mathutils
import numpy as np
try:
    import imageio
except ImportError:
    import pip
    pip.main(["install", "imageio[ffmpeg]", "--user"])
    import imageio

VIEWS = 72
RESOLUTION = 800
FORMAT = "PNG"
RENDER_SCENE_FLOW = True
PI = 3.14
SCALE = 2.2
INPUT_FN = "assets/0.obj"
RESULTS_PATH = f"results-views{VIEWS}"

FIX_SEED = True
if FIX_SEED:
    import random
    random.seed(2)
    np.random.seed(2)

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def set_camera(bpy_cam, angle=PI/3, W=600, H=500):
    """TODO: replace with setting by intrinsics """
    bpy_cam.angle = angle
    bpy_scene = bpy.context.scene
    bpy_scene.render.resolution_x = W
    bpy_scene.render.resolution_y = H

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras "-Z" and use its "Y" as up
    rot_quat = direction.to_track_quat("-Z", "Y")
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

def get_calibration_matrix_K_from_blender(camd):
    """ From DeformingThings4D """
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == "VERTICAL"):
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
        s_u = s_v/pixel_aspect_ratio
    else: # "HORIZONTAL" and "AUTO"
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = s_u/pixel_aspect_ratio
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels
    K = mathutils.Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K

def opencv_to_blender(T):
    """T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0,  0, 1)))
    return np.matmul(T,origin)

def blender_to_opencv(T):
    transform = np.array(((1, 0, 0, 0),
              (0, -1, 0, 0),
              (0, 0, -1, 0),
              (0, 0, 0, 1)))
    return np.matmul(T,transform)

def set_cycles_renderer(scene: bpy.types.Scene,
                        camera_object: bpy.types.Object,
                        num_samples: int,
                        use_denoising: bool = True,
                        use_motion_blur: bool = False,
                        use_transparent_bg: bool = False,
                        prefer_cuda_use: bool = True,
                        use_adaptive_sampling: bool = False) -> None:
    scene.camera = camera_object

    scene.render.image_settings.file_format = "PNG"
    scene.render.engine = "CYCLES"
    scene.render.use_motion_blur = use_motion_blur

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.use_adaptive_sampling = use_adaptive_sampling
    scene.cycles.samples = num_samples

    # Enable GPU acceleration
    # Source - https://blender.stackexchange.com/a/196702
    if prefer_cuda_use:
        bpy.context.scene.cycles.device = "GPU"

        # Change the preference setting
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

    # Call get_devices() to let Blender detects GPU device (if any)
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # Let Blender use all available devices, include GPU and CPU
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1

    # Display the devices to be used for rendering
    print("----")
    print("The following devices will be used for path tracing:")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        print("- {}".format(d["name"]))
    print("----")

def generate_video(image_list, output_file, fps=18):
    with imageio.get_writer(output_file, fps=fps) as writer:
        for image in image_list:
            writer.append_data(image)
            
def main():
    fp = bpy.path.abspath(f"//{RESULTS_PATH}")
    os.makedirs(fp, exist_ok=True)

    # render params
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.image_settings.file_format = str(FORMAT)

    # create collection for objects not to render with background
    # delete default cube
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Cube"].select_set(state=True)
    bpy.ops.object.delete(use_global=False)

    # light
    light_ = bpy.data.lights["Light"]
    light_.type = "SUN"
    light_.use_shadow = False
    # Possibly disable specular shading:
    light_.specular_factor = 1.0
    light_.energy = 5

    # add another light source so stuff facing away from light is not completely dark
    bpy.ops.object.light_add(type="SUN")
    light_2 = bpy.data.lights["Sun"]
    light_2.use_shadow = False
    light_2.specular_factor = 1.0
    light_2.energy = 5
    # bpy.data.objects["Sun"].location = (0, 100, 0)
    bpy.data.objects["Sun"].rotation_euler = bpy.data.objects["Light"].rotation_euler
    bpy.data.objects["Sun"].rotation_euler[0] += 180

    # rendering
    bpy.context.scene.render.use_placeholder = False
    # Background
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = RESOLUTION
    bpy.context.scene.render.resolution_y = RESOLUTION
    bpy.context.scene.render.resolution_percentage = 100

    # set up camera
    cam = bpy.context.scene.objects["Camera"]
    cam.location = mathutils.Vector((2, -2, 1))
    look_at_point = mathutils.Vector((0, 0, 1)) # need to compute this for optimal view point
    look_at(cam, look_at_point)    
    set_camera(cam.data, angle=PI/3, W=RESOLUTION, H=RESOLUTION)
    bpy.context.view_layer.update() #update camera params

    # dump intrinsics & extrinsics
    K = get_calibration_matrix_K_from_blender(cam.data)
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    np.savetxt(f"{fp}/cam_intr.txt" , np.array(K))
    cam_blender = np.array(cam.matrix_world)
    cam_opencv = blender_to_opencv(cam_blender)
    np.savetxt(f"{fp}/cam_ext.txt" , cam_opencv)
    
    # compute ray
    u, v = np.meshgrid(range(RESOLUTION), range(RESOLUTION))
    u = u.reshape(-1)
    v = v.reshape(-1)
    pix_position = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    cam_rotation = cam_opencv[:3, :3]
    pix_position = np.matmul(cam_rotation, pix_position.transpose()).transpose()
    ray_direction = pix_position / np.linalg.norm(pix_position, axis=1, keepdims=True)
    ray_origin = cam_opencv[:3, 3:].transpose()
    
    # import textured mesh
    bpy.ops.wm.obj_import(filepath=INPUT_FN)
    mesh_obj = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")
    bpy.context.view_layer.objects.active = mesh_obj    
    
    # scale
    factor = max(mesh_obj.dimensions[0], mesh_obj.dimensions[1], mesh_obj.dimensions[2]) / SCALE
    mesh_obj.scale[0] /= factor
    mesh_obj.scale[1] /= factor
    mesh_obj.scale[2] /= factor    
    bpy.ops.object.transform_apply(scale=True)
    vert_vector0 = np.stack([np.array(item.co) for item in mesh_obj.data.vertices])
    faces = mesh_obj.data.polygons

    for i in range(VIEWS+1):
        stepsize = 360.0 / VIEWS if i > 0 else 0
        print(f"Rendering view {i} at stepsize {stepsize}")
        # render
        bpy.context.scene.render.filepath = fp + "/%06d" % i
        bpy.ops.render.render(write_still=True)
                
        # update geometry (rotation)
        bm = bmesh.new()
        bm.from_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for i in range(len(bm.verts)):   
            # rotation along Z axis
            euler_rotation = mathutils.Euler((0, 0, math.radians(stepsize)), "XYZ") # XYZ is default
            R = euler_rotation.to_matrix() # 3 x 3 rotation matrix
            bm.verts[i].co = R @ bm.verts[i].co
        bm.to_mesh(mesh_obj.data)
        mesh_obj.data.update()
        
        # explicitly cast rays to get point cloud
        ray_begin_local = mesh_obj.matrix_world.inverted() @ mathutils.Vector(ray_origin[0])
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bvhtree = mathutils.bvhtree.BVHTree.FromObject(mesh_obj, depsgraph)
        pcl = np.zeros_like(ray_direction)
        sflow = np.zeros_like(ray_direction)
        vert_vector_cur = np.stack([np.array(item.co) for item in mesh_obj.data.vertices])
        for _j in range(ray_direction.shape[0]):
            # cast ray:
            #   - position: where the ray hits the triangle
            #   - faceID: the index of the triangle which is hit by the ray
            position, _, faceID, _  = bvhtree.ray_cast(ray_begin_local, mathutils.Vector(ray_direction[_j]), 200)
            if position: # hit a triangle
                pcl[_j]= mathutils.Matrix(cam_opencv).inverted() @ mesh_obj.matrix_world @ position
                if RENDER_SCENE_FLOW:
                    # get vertex id for current face
                    vert_index = [v for v in faces[faceID].vertices]
                    # get the 3D position of the vertex
                    vert_vector = [mesh_obj.data.vertices[v].co for v in vert_index]
                    # compute the barycentric weights
                    weights = np.array(mathutils.interpolate.poly_3d_calc(vert_vector, position))
                    # compute the flow vector
                    flow_vector = (vert_vector_cur[vert_index] - vert_vector0[vert_index]) * weights.reshape([3,1])
                    # average over the 3 vertices
                    sflow[_j] = flow_vector.sum(axis=0)
                    
        # dump depth
        depth = pcl[:,2].reshape((RESOLUTION, RESOLUTION))
        depth_1mm = (depth * 1000).astype(np.uint16) #  resolution 1mm
        np.save(f"{bpy.context.scene.render.filepath}_depth.npy", depth_1mm)
        depth_vis = (depth / depth.max() * 255).astype(np.uint8)
        imageio.imwrite(f"{bpy.context.scene.render.filepath}_depth.png", depth_vis)
        if RENDER_SCENE_FLOW:
            # rotate to camera coordinate system (opencv)
            sflow = np.matmul(np.linalg.inv(cam_opencv[:3, :3]), sflow.transpose()).transpose() 
            sflow = sflow.reshape((RESOLUTION, RESOLUTION, 3)).astype(np.float32)
            imageio.imwrite(f"{bpy.context.scene.render.filepath}_sflow.exr", sflow)
            sflow = (sflow - sflow.min()) / (sflow.max() - sflow.min()) * 255
            imageio.imwrite(f"{bpy.context.scene.render.filepath}_sflow.png", sflow.astype(np.uint8))

    # generate video for visualization
    renderings = [imageio.imread(f"{fp}/{i:06d}.png") for i in range(VIEWS)]
    depth_vis = [imageio.imread(f"{fp}/{i:06d}_depth.png") for i in range(VIEWS)]
    flow_vis = [imageio.imread(f"{fp}/{i:06d}_sflow.png") for i in range(VIEWS)]
    concat_vis = [np.concatenate([renderings[i][:,:,:3], np.tile(depth_vis[i].reshape(RESOLUTION, RESOLUTION, 1), (1,1,3)), flow_vis[i]], axis=1) for i in range(VIEWS)]
    generate_video(concat_vis, f"{fp}/concat_vis.mp4")
    
if __name__ == "__main__":
    main()