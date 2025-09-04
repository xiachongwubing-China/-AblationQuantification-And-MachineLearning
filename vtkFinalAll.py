import numpy as np
import nibabel as nib
import vtk
from scipy import ndimage
import numpy as np
def load_and_smooth_mask(file_path, color, opacity,
                         label_value=0.5, smooth_iterations=80, pass_band=0.05, fill_hole_size=1000.0):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_path)
    reader.Update()

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputConnection(reader.GetOutputPort())
    contour.SetValue(0, label_value)
    contour.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(smooth_iterations)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(smoother.GetOutputPort())
    cleaner.Update()

    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputConnection(cleaner.GetOutputPort())
    fill_holes.SetHoleSize(fill_hole_size)
    fill_holes.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(fill_holes.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SetFeatureAngle(120.0)
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetInterpolationToPhong()

    return actor

def compute_mask_center(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    coords = np.array(np.where(data > 0)).T
    if coords.shape[0] == 0:
        return np.array([0,0,0])
    spacing = img.header.get_zooms()
    origin = img.affine[:3, 3]
    center_voxel = coords.mean(axis=0)
    return center_voxel * spacing + origin

def add_arrow_with_label(renderer, start_point, end_point, label_text, color=(1,1,1)):
    arrow_source = vtk.vtkArrowSource()
    arrow_source.SetTipLength(0.1)
    arrow_source.SetTipRadius(0.03)
    arrow_source.SetShaftRadius(0.01)

    direction = np.array(end_point) - np.array(start_point)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return
    direction = direction / norm

    transform = vtk.vtkTransform()
    transform.Translate(start_point)

    arbitrary = vtk.vtkTransform()
    angle = np.degrees(np.arccos(np.clip(direction.dot([1,0,0]), -1.0, 1.0)))
    axis = np.cross([1,0,0], direction)
    if np.linalg.norm(axis) > 1e-6:
        transform.RotateWXYZ(angle, *axis)
    transform.Scale(norm, norm, norm)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(arrow_source.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    renderer.AddActor(actor)

    text_source = vtk.vtkVectorText()
    text_source.SetText(label_text)
    text_mapper = vtk.vtkPolyDataMapper()
    text_mapper.SetInputConnection(text_source.GetOutputPort())
    text_actor = vtk.vtkFollower()
    text_actor.SetMapper(text_mapper)
    text_actor.SetScale(5, 5, 5)
    text_actor.GetProperty().SetColor(color)
    text_actor.SetPosition(*start_point)
    text_actor.SetCamera(renderer.GetActiveCamera())
    renderer.AddActor(text_actor)


def process_min_thickness_region(region_info):
    """Find the area of ​​minimum wall thickness and return to its center"""
    if region_info:
        min_thickness_region = min(region_info, key=lambda x: x['thickness_mm'])
        return min_thickness_region['center_mm']
    return np.array([0, 0, 0])

def load_masks(path1, path2):
    mask1_img = nib.load(path1)
    mask2_img = nib.load(path2)
    mask1_data = mask1_img.get_fdata().astype(bool)
    mask2_data = mask2_img.get_fdata().astype(bool)
    spacing = mask1_img.header.get_zooms()
    origin = mask1_img.affine[:3, 3]
    return mask1_data, mask2_data, spacing, origin

def boolean_subtraction(mask1, mask2):
    return mask1 & (~mask2)

def label_regions(subtracted_mask):
    labeled_mask, num_features = ndimage.label(subtracted_mask)
    return labeled_mask, num_features

def calculate_wall_thickness(mask, spacing):
    distance_transform = ndimage.distance_transform_edt(mask)  

    inverted_mask = np.logical_not(mask)  
    inside_distance_transform = ndimage.distance_transform_edt(inverted_mask)  

    wall_thickness = distance_transform - inside_distance_transform  

    min_thickness_voxel = np.min(wall_thickness[mask])
    min_thickness_mm = min_thickness_voxel * spacing[2]  

    return min_thickness_mm


def process_regions(labeled_mask, num_features, spacing, origin):
    region_info = []
    for region_id in range(1, num_features + 1):
        single_region = labeled_mask == region_id
        coords = np.array(np.where(single_region)).T
        if len(coords) < 10:
            continue
        thickness_mm = calculate_wall_thickness(single_region, spacing)
        center_mm = coords.mean(axis=0) * spacing + origin
        region_info.append({
            'id': region_id,
            'center_mm': center_mm,
            'thickness_mm': thickness_mm,
            'points_voxel': coords
        })
    return region_info

def visualize_all(renderer, region_info, spacing, origin):
    color = [0.0, 0.5, 1.0]
    for info in region_info:
        coords_voxel = info['points_voxel']
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for pt in coords_voxel:
            pos = pt * spacing + origin
            pid = points.InsertNextPoint(*pos)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetPointSize(5)
        actor.GetProperty().SetOpacity(0.9)
        renderer.AddActor(actor)
def find_surface_point_along_direction(mask_data, spacing, origin, start_point, direction, step_size=0.5, max_steps=1000):
    start_voxel = np.round((np.array(start_point) - origin) / spacing).astype(int)

    dir_normalized = np.array(direction) / np.linalg.norm(direction)

    step_in_voxel = (step_size / np.array(spacing)) * dir_normalized

    current_voxel = start_voxel.astype(float)

    for _ in range(max_steps):
        current_voxel += step_in_voxel
        idx = np.round(current_voxel).astype(int)

        if any(idx < 0) or any(idx >= np.array(mask_data.shape)):
            return None

        if mask_data[tuple(idx)]:
            surface_point = idx * spacing + origin
            return surface_point

    return None  
def main():
    mask1_path = 'OriginalImage_res.nii.gz'
    mask2_path = 'Seglabel.nii.gz'
    mask3_path = 'lunglabel.nii.gz'

    mask1, mask2, spacing, origin = load_masks(mask1_path, mask2_path)
    subtracted = boolean_subtraction(mask2, mask1)
    labeled_mask, num_features = label_regions(subtracted)
    region_info = process_regions(labeled_mask, num_features, spacing, origin)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    actors = [
        load_and_smooth_mask(mask1_path, (1, 0, 0), 0.3),
        load_and_smooth_mask(mask2_path, (0, 1, 0), 0.6),
        load_and_smooth_mask(mask3_path, (1, 0.41, 0.7), 0.2),
    ]
    for actor in actors:
        renderer.AddActor(actor)

    visualize_all(renderer, region_info, spacing, origin)

    gtr_pos = compute_mask_center(mask1_path)
    ptz_pos = compute_mask_center(mask2_path)  

    arrow_offset_gtr = np.array([60, 30, 10])     
    arrow_offset_ptz = np.array([-50, 40, 20])    
    arrow_offset_thick = np.array([30, -50, 30])  


    add_arrow_with_label(renderer, gtr_pos + arrow_offset_gtr, gtr_pos, "GTR", (1, 0, 0))

    start_point = ptz_pos + arrow_offset_ptz
    direction = ptz_pos - start_point  

    surface_point = find_surface_point_along_direction(
        mask2, spacing, origin, start_point, direction
    )

    if surface_point is not None:
        print("Find PTZ surface points：", surface_point)
        add_arrow_with_label(renderer,
                              start_point,
                              surface_point,
                              "PTZ",
                              (0, 1, 0))
    else:
        print("Failed to find PTZ surface point")

    if region_info:
        min_thickness_region = min(region_info, key=lambda x: x['thickness_mm'])
        mam_center = min_thickness_region['center_mm']
        thickness = min_thickness_region['thickness_mm']

        arrow_offset_thick = np.array([-20, -50, -30]) 
        start_point = mam_center + arrow_offset_thick  

        direction = mam_center - start_point

        unit_direction = direction / np.linalg.norm(direction)

        extension_distance = 8.0  # mm
        extended_end_point = mam_center + unit_direction * extension_distance

        add_arrow_with_label(renderer,
                            start_point,
                            extended_end_point,
                            f"MAM = {thickness :.2f}mm",
                            (1, 1, 0))  
    renderer.SetBackground(0.2, 0.2, 0.2)
    render_window.SetSize(1200, 1000)
    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    main()
