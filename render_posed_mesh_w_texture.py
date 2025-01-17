from PIL import Image
import cv2
import numpy as np
import torch
import smplx
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    TexturesUV,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings, 
    BlendParams, 
    Materials
)
from pytorch3d.structures import Meshes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v, f, aux = load_obj("/data/smplx_models/smpl_uv.obj")
    verts_uv = aux.verts_uvs
    texture_image_path = "/data/smpl_models/smpl/m_01_alb.002.png"
    texture_image = Image.open(texture_image_path)
    texture_image = texture_image.convert("RGB")
    texture_image = np.array(texture_image) / 255.0
    texture_image = torch.tensor(texture_image, dtype=torch.float32)[None]

    # smpl_mesh = load_objs_as_meshes(["/data/smplx_models/smpl_uv.obj"], device=device)

    faces = f.verts_idx
    faces_uvs = f.textures_idx
    # verts_uv = torch.cat((verts_uv[:, [0]], 1-verts_uv[:, [1]]), dim=1)
    
    texture = TexturesUV(
        maps=texture_image, 
        faces_uvs=faces_uvs[None], 
        verts_uvs=verts_uv[None],
    )

    # Path to the folder containing the SMPL-X model files
    model_folder = "/data/smplx_models/models"

    # Specify the model type ('smplx', 'smpl', 'smplh', or 'mano')
    model_type = 'smpl'

    # Initialize the SMPL-X model
    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender='male',
        use_pca=False,
        batch_size=1
    )

    # Set random body pose, betas, and global orientation
    body_pose = torch.zeros(1, 23*3)  # 21 joints * 3 (axis-angle representation)
    betas = torch.zeros(1, 10)      # 10 shape coefficients
    global_orient = torch.zeros(1, 3)  # Global rotation

    # Forward pass through the model
    output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        return_verts=True
    )
    verts = output.vertices
    # faces = model.faces.astype(int)

    meshes = Meshes(
        verts=verts.to(device), 
        faces=faces.unsqueeze(0).to(device),
        textures=texture.to(device),
    )

    # Set up renderer
    # Set up materials, lights, and cameras
    materials = Materials(device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0)
    lights = PointLights(
        device=device,
        location=[[0.0, 1.0, 1.0]],
        ambient_color=((0.8, 0.8, 0.8),),
        diffuse_color=((0.9, 0.9, 0.9),),
        specular_color=((1.0, 1.0, 1.0),),
    )
    R, T = look_at_view_transform(dist=2.7, elev=30, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,  # Naive rasterization to avoid overflow
        max_faces_per_bin=50000  # Increase face limit per bin if using larger meshes
    )

    blend_params = BlendParams(background_color=(0, 0, 0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),

        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            blend_params=blend_params,
            lights=lights,
            materials=materials
        )
    )

    img = renderer(meshes)
    
    img = img[0, ..., :3].detach().cpu().numpy()*255
    cv2.imwrite("result.jpg", img[..., ::-1])

if __name__ == '__main__':
    main()