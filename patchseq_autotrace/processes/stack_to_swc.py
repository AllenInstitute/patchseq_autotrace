import numpy as np
import pandas as pd
from collections import deque
import tifffile as tif
import itertools
from operator import add
import os
from collections import defaultdict
import cv2
import shutil
from scipy.spatial import distance
from scipy.ndimage import label, morphology, generate_binary_structure
from patchseq_autotrace.utils import get_63x_soma_coords, get_tifs, natural_sort
from patchseq_autotrace import __version__ as autotrace_code_version

def assign_parent_child_relation(start_node, start_nodes_parent, parent_dict, neighbors_dict):
    """
    Starting at a given leaf node of a connected component, walk over the structure using breadth
    first to assign parent-child node relationships. This function will fill out the parent_dict
    variable for a given connected component

    :param start_node: tuple, (x,y,z) coordinate to start at in walking connected components
    :param start_nodes_parent: int, node id for the parent node. Either soma or -1
    :param parent_dict: dict, {tuple:tuple} representing child_node:parent_node
    :param neighbors_dict: dict, {tuple:list} represents the neighboring coordinates for each coordinate
    :return: None, populates parent dict
    """
    # this function uses BFS to assign parent child relationships in the neighbors dict
    parent_dict[start_node] = start_nodes_parent
    queue = deque([start_node])
    while len(queue) > 0:
        current_node = queue.popleft()
        my_connections = neighbors_dict[current_node]
        for node in my_connections:
            if node not in parent_dict:
                # print('Assigning node {} to be the child of {}'.format(node_dict[node],node_dict[current_node]))
                parent_dict[node] = current_node
                queue.append(node)
            # else:
            # p = 'Initial start node' if parent_dict[node] == start_nodes_parent else str([parent_dict[node]])
            # print('{} already has a parent {}'.format(node_dict[node], p))


def consolidate_conn_components(connected_components_to_merge):
    """
    This function will merge connected components that were interrupted by chunking of the entire image stack. In
    visual above, connected component 34 and 60 should merge into one connected component. This code will handle all
    merge scenarios (across various chunk indices, many components merge to one and vice versa). See visual below:

    / and \ represent neuron segment

                /     (connected component 22)
              /  \
    -------------------- (chunk index)
           /       \
         /          \       (connected component 38 and 66)

    :param connected_components_to_merge: defaultdict(set), keys represent connected component labels and values are
    the set of connected component labels that should merge into the key. E.g. {22:{38,66}, 38:{22}, 66:{22}} where
    integers represent the connected component label.

    :return: dict, keys represent connected components and values are the set of connected component a key will be
    reassigned to. An empty set indicates that a given connected component is not changing label.
     Continuing with example given above, return would be: {22:{}, 38:{22}, 66:{22}}
    """
    pre_qc = set()
    all_nodes = {}
    for k, v in connected_components_to_merge.items():
        pre_qc.add(k)
        all_nodes[k] = set()
        for vv in v:
            pre_qc.add(vv)
            all_nodes[vv] = set()

    nodes_to_remove = set()
    for start_node in connected_components_to_merge.keys():
        rename_count = 0
        # print('')
        # print('Start Node {} assignment = {}'.format(start_node, all_nodes[start_node]))
        if all_nodes[start_node] == set():
            # print('Starting at {}'.format(start_node))
            queue = deque([start_node])
            start_value = start_node
            back_track_log = set()
            while len(queue) > 0:

                current_node = queue.popleft()

                if all_nodes[current_node] == set():
                    # print('assigning current node {} to {}'.format(current_node, start_value))
                    back_track_log.add(current_node)
                    all_nodes[current_node].add(start_value)
                    if current_node in connected_components_to_merge.keys():
                        # print('appending children to the queue')
                        [queue.append(x) for x in connected_components_to_merge[current_node]]
                else:
                    rename_count += 1
                    if rename_count < 2:
                        nodes_to_remove.add(start_node)
                        # print('Backtracking when i got to node {}'.format(current_node))
                        start_value = next(iter(all_nodes[current_node]))
                        # print('Updating the value to be {}'.format(start_value))
                        for node in back_track_log:
                            # print('Going back to update node {} to {}'.format(node, start_value))
                            all_nodes[node].clear()
                            all_nodes[node].add(start_value)

                    else:
                        # need to remove all nodes that have this already assigned value
                        # and updated them to the start_value assigned on line 36
                        value_to_remove = next(iter(all_nodes[current_node]))
                        # print('Found {} already labeled node at {}. Its label = {}'.format(rename_count, current_node,
                        #                                                                    value_to_remove))
                        nodes_to_push_rename = [k for k, v in all_nodes.items() if value_to_remove in v]

                        for node in nodes_to_push_rename:
                            all_nodes[node].clear()
                            all_nodes[node].add(start_value)
                            if node in connected_components_to_merge.keys():  # cant remove it from the keys if its a leaf
                                nodes_to_remove.add(start_node)

        # else:
        #     print('was going to analyze {} but its already assigned to {}'.format(start_node, all_nodes[start_node]))

    my_dict = defaultdict(set)
    for k, v in all_nodes.items():
        if k != next(iter(v)):
            my_dict[next(iter(v))].add(k)

    post_qc = set()
    for k, v in my_dict.items():
        post_qc.add(k)
        for vv in v:
            post_qc.add(vv)

    for i in pre_qc:
        if i not in post_qc:
            print('Node {} is in the input dict but not output'.format(i))

    return my_dict


def get_soma_xyz(max_intensity_proj_ch1_pth, yz_mip_pth, specimen_id, min_pixel_size_of_soma=500):
    """
    Our soma segmentation will segment all cells in an image stack, not just our cell of interest. Also, things like
    soma leakage occurs which causes many soma dots to appear in the max intensity projection. This function will take
    max intensity projection in xy and best identify which soma is the one for our cell of interest.
    :param max_intensity_proj_ch1_pth:
    :param specimen_id:
    :return:
    """
    no_soma = False
    if not os.path.exists(max_intensity_proj_ch1_pth):
        no_soma = True
        centroid = (0, 0, 0)
        connection_threshold = 0

        return centroid, connection_threshold, no_soma

    ch1_mip = tif.imread(max_intensity_proj_ch1_pth)

    # if we can get 63x soma coordinates from LIMS we will use those as our x and y
    # then we only need to solve for z
    xs_63x, ys_63x = get_63x_soma_coords(specimen_id)
    spacer = 300

    if (xs_63x != None) and (ys_63x != None):
        avg_x = np.mean(xs_63x)
        avg_y = np.mean(ys_63x)

        # no soma is going to be bigger than 600 x 600 pixels, right?
        x0, x1 = int(avg_x - spacer), int(avg_x + spacer)
        y0, y1 = int(avg_y - spacer), int(avg_y + spacer)
        # ROI max
        mip_max = ch1_mip[y0:y1, x0:x1].max()
        # zero out non ROI
        ch1_mip[0:y0, :] = 0
        ch1_mip[y1:, :] = 0
        ch1_mip[:, 0:x0] = 0
        ch1_mip[:, x1:] = 0
        # we're not done yet, we still need to do some thresholding/erosion.

    else:
        mip_max = ch1_mip.max()

    # create a minimum signal value for our ch1 segmentation
    if mip_max < 100:
        cutoff = int(mip_max * 0.15)
    else:
        cutoff = int(mip_max * 0.3)

    # Find the connected component label of our soma of interest
    ch1_mip[ch1_mip < cutoff] = 0
    struct = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    conn_comps, num_comps = label(ch1_mip, structure=struct)

    if (xs_63x != None) and (ys_63x != None):
        # we can just use the 63x lims data
        soma_cc_label = conn_comps[int(avg_y), int(avg_x)]

    else:
        # ASSUMPTION: The soma of interest is the center most soma found

        img_shape = ch1_mip.shape
        cc_count = 0
        list_of_cc_labels = []
        for i in range(num_comps + 1):
            if i != 0:
                index = np.where(conn_comps == i)
                if len(index[0]) > min_pixel_size_of_soma:  # want at least 500 pixels for our soma
                    cc_count += 1
                    list_of_cc_labels.append(i)

        # Threshold was too aggressive
        if list_of_cc_labels == []:
            for i in range(num_comps + 1):
                if i != 0:
                    index = np.where(conn_comps == i)
                    cc_count += 1
                    list_of_cc_labels.append(i)

        # if its still empty, no connected components found
        if list_of_cc_labels == []:
            soma_cc_label = 0

        # find the connected component that's center is closest to the center of the image
        middle_of_image_x = int(img_shape[1] / 2)
        middle_of_image_y = int(img_shape[0] / 2)
        closest_distance = np.inf
        for cc_ind in list_of_cc_labels:
            index = np.where(conn_comps == cc_ind)
            conn_comp_xs = index[1]
            conn_comp_ys = index[0]
            cc_mean_x = np.mean(conn_comp_xs)
            cc_mean_y = np.mean(conn_comp_ys)

            this_cc_dist_to_center = distance.euclidean((cc_mean_y, cc_mean_x),
                                                        (middle_of_image_y, middle_of_image_x))

            if this_cc_dist_to_center < closest_distance:
                soma_cc_label = cc_ind
                closest_distance = this_cc_dist_to_center

    # The x and y values of our decided soma
    chosen_ys, chosen_xs = np.where(conn_comps == soma_cc_label)
    print("connected component label: {}".format(soma_cc_label))
    print("CC Shape: {},{}".format(len(chosen_ys), len(chosen_xs)))

    # binarize mip so only chosen soma cc location remains
    conn_comps[conn_comps != soma_cc_label] = 0
    conn_comps[conn_comps == soma_cc_label] = 1

    # check that this chosen connected component isn't a massive ball of soma fill
    # here we will erode the soma conn comp until it fits into our acceptable size
    max_dx_soma_cc = max(chosen_xs) - min(chosen_xs)
    max_dy_soma_cc = max(chosen_ys) - min(chosen_ys)

    soma_size_thresh = 2 * spacer - 20
    erosion_stopping_thresh = 50
    if (max_dx_soma_cc > soma_size_thresh) or (max_dy_soma_cc > soma_size_thresh):

        # background was identified as cc label. Could be that image doesnt match w/ lims or segmentation
        # did not do a good job of capturing soma.
        if soma_cc_label != 0:

            print("Going to erode the soma connected component a bit because it exceeds our size threshold")
            eroded_conn_comp = conn_comps.copy()
            stopping_condition = False
            while not stopping_condition:
                eroded_conn_comp = morphology.binary_erosion(eroded_conn_comp).astype(eroded_conn_comp.dtype)
                temp_xs, temp_ys = np.where(eroded_conn_comp == 1)
                temp_dx = max(temp_xs) - min(temp_xs)
                temp_dy = max(temp_ys) - min(temp_ys)

                if (temp_dx < soma_size_thresh) and (temp_dy < soma_size_thresh):
                    print("   temp dx {} and dy {} is less than our threshold {}".format(temp_dx, temp_dy,
                                                                                         soma_size_thresh))
                    stopping_condition = True
                elif (temp_dx < erosion_stopping_thresh) or (temp_dy < erosion_stopping_thresh):
                    # we dont want to completely erode away so we will stop
                    print("   Woah there. One of our dimensions has been eroded to our erosion threhsold")
                    stopping_condition = True

            conn_comps = eroded_conn_comp.copy()
            chosen_ys, chosen_xs = np.where(conn_comps == 1)

    min_y, max_y = int(min(chosen_ys)), int(max(chosen_ys))
    min_x, max_x = int(min(chosen_xs)), int(max(chosen_xs))

    # Solve for soma z
    yz_mip = tif.imread(yz_mip_pth)
    # zero out any irrelevant y values
    yz_mip[:min_y, :] = 0
    yz_mip[max_y:, :] = 0

    # thresholding
    yz_mip_roi_max = yz_mip.max()
    if yz_mip_roi_max < 100:
        yz_cutoff = int(yz_mip_roi_max * 0.15)
    else:
        yz_cutoff = int(yz_mip_roi_max * 0.3)

    # iterate and choose largest connected component in this y-column
    yz_mip[yz_mip < yz_cutoff] = 0
    yz_conn_comps, yz_num_comps = label(yz_mip)
    biggest_component_size = 0
    for i in range(yz_num_comps + 1):
        if i != 0:
            index = np.where(yz_conn_comps == i)
            size_comp = len(index[0])
            if size_comp > biggest_component_size:
                biggest_component_size = size_comp
                yz_conn_comp_label = i

    yz_conn_comps[yz_conn_comps != yz_conn_comp_label] = 0
    yz_conn_comps[yz_conn_comps == yz_conn_comp_label] = 1

    chosen_zs, _ = np.where(yz_conn_comps == 1)

    centroid = (np.mean(chosen_xs), np.mean(chosen_ys), np.mean(chosen_zs))

    mean_diameter = ((max_y - min_y) + (max_x - min_x)) / 2
    mean_radius = mean_diameter / 2
    connection_threshold = mean_radius * 1.25

    return centroid, connection_threshold, mean_radius, no_soma


def skeleton_to_swc(specimen_dir, model_and_version, max_stack_size=7000000000):
    sp_id = os.path.basename(os.path.abspath(specimen_dir))
    print('Starting To Process {}'.format(sp_id))

    skeleton_dir = os.path.join(specimen_dir, 'Skeleton')

    skeleton_labels_file = os.path.join(specimen_dir, 'Segmentation_skeleton_labeled.csv')

    # Calculate how many files to load as not to exceed memory limit per iteration
    filelist = natural_sort(get_tifs(skeleton_dir))
    filename = os.path.join(skeleton_dir, filelist[0])
    img = tif.imread(filename)
    cell_stack_size = len(filelist), img.shape[0], img.shape[1]
    cell_stack_memory = cell_stack_size[0] * cell_stack_size[1] * cell_stack_size[2]
    print('cell_stack_size (z,y,x):', cell_stack_size, cell_stack_memory)
    # if cell stack memory>max_stack_size need to split
    num_parts = int(np.ceil(cell_stack_memory / max_stack_size))
    print('num_parts:', num_parts)

    idx = np.append(np.arange(0, cell_stack_size[0], int(np.ceil(cell_stack_size[0] / num_parts))),
                    cell_stack_size[0] + 1)
    shared_slices = idx[1:-1]
    both_sides_of_slices = np.append(shared_slices, shared_slices - 1)

    # Initialize variables before begining chunk looping
    connected_components_on_border = {}
    for j in both_sides_of_slices:
        connected_components_on_border[j] = []
    previous_cc_count = 0
    slice_count = 0
    full_neighbors_dict = {}
    node_component_label_dict = {}
    cc_dict = {}

    for i in range(num_parts):
        print('At Part {}'.format(i))
        idx1 = idx[i]
        idx2 = idx[i + 1]
        filesublist = filelist[idx1:idx2]
        # print('part ', i, idx1, idx2, len(filesublist))

        # load stack and run connected components
        cv_stack = []
        for f in filesublist:
            filename = os.path.join(skeleton_dir, f)
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            cv_stack.append(img)
        three_d_array = np.stack(cv_stack)
        struct = generate_binary_structure(3, 3)
        labels_out, _ = label(three_d_array, structure=struct)

        current_number_of_components = np.max(labels_out)
        # print('There are {} CCs in this stack of images'.format(current_number_of_components))

        # Create range for connected components across all chunks of image stack
        if previous_cc_count == 0:
            cc_range = range(1, current_number_of_components + 1)
        else:
            cc_range = range(previous_cc_count + 1, previous_cc_count + 1 + current_number_of_components)

        for cc in cc_range:
            cc_dict[cc] = {'X': [], 'Y': [], 'Z': []}

        # Load each image slice by slice so that we can get coordinates to the connected components
        for single_image in labels_out:
            single_image_unique_labels = np.unique(single_image)  # return indices and ignore 0
            for unique_label in single_image_unique_labels:
                if unique_label != 0:
                    indices = np.where(single_image == unique_label)
                    conn_comp_apriori_num = unique_label + previous_cc_count
                    [cc_dict[conn_comp_apriori_num]['Y'].append(coord) for coord in indices[0]]
                    [cc_dict[conn_comp_apriori_num]['X'].append(coord) for coord in indices[1]]
                    [cc_dict[conn_comp_apriori_num]['Z'].append(x) for x in [slice_count] * len(indices[1])]

                    if slice_count in both_sides_of_slices:
                        connected_components_on_border[slice_count].append(conn_comp_apriori_num)

            slice_count += 1

        ################################################################################################################
        # To turn a connected component into a SWC file, we need to keep track of coordinate neighbors (e.g. this
        # voxel (1,1,1) may have neighbors at (1,0,0) and (2,1,1), or it may be a leaf node and only have one neighbor
        # Here we are using neighbors_dict to keep track of those relationships
        ################################################################################################################

        for conn_comp in cc_range:
            coord_values = cc_dict[conn_comp]
            component_coordinates = np.array([coord_values['X'], coord_values['Y'], coord_values['Z']]).T

            # Making a node dictionary for this con comp so we can lookup in the 26 node check step
            node_dict = {}
            count = 0
            for c in component_coordinates:
                count += 1
                node_dict[tuple(c)] = count
                node_component_label_dict[tuple(c)] = conn_comp

            # 26 nodes to check in defining neighbors dict
            movement_vectors = ([p for p in itertools.product([0, 1, -1], repeat=3)])
            neighbors_dict = {}
            for node in component_coordinates:

                node_neighbors = []
                for vect in movement_vectors:
                    node_to_check = tuple(list(map(add, tuple(node), vect)))
                    if node_to_check in node_dict.keys():
                        node_neighbors.append(node_to_check)

                # remove myself from my node neightbors list
                node_neighbors = set([x for x in node_neighbors if x != tuple(node)])
                neighbors_dict[tuple(node)] = node_neighbors
                full_neighbors_dict[conn_comp] = neighbors_dict

        previous_cc_count += current_number_of_components

    ################################################################################################################
    # All image chunks have been loaded and full neighbors dict is constructed. Now, since we chunked our image, we need
    # to stitch connected components across the chunk indices and consolidate connected component labels. Here we use
    # left and right to denote deeper (left, greater z value) and more superficial (chunks, lower z values) of our image
    # stack.
    ################################################################################################################
    print('Merging Conn Components across chunk indexes')

    # Initializing Nodes on either side of slice boundary
    nodes_to_left_of_boundary = {}
    for x in shared_slices - 1:
        nodes_to_left_of_boundary[x] = defaultdict(list)

    nodes_to_right_of_boundary = {}
    for x in shared_slices:
        nodes_to_right_of_boundary[x] = defaultdict(list)

    # assigning nodes only with z value on edge to left or right side
    for key, val in connected_components_on_border.items():
        for con_comp_label in val:
            coord_values = full_neighbors_dict[con_comp_label].keys()
            for coord in coord_values:
                z = coord[-1]
                if z == key:
                    if z in shared_slices - 1:
                        nodes_to_left_of_boundary[key][con_comp_label].append(tuple(coord))
                    else:
                        nodes_to_right_of_boundary[key][(tuple(coord))] = con_comp_label

    ################################################################################################################
    # Check the 26 boxes surrounding each node that lives on the left side
    # Update full neighbors dictionary
    # Create dictionary of conn components that need to merge across slize index
    ################################################################################################################

    movement_vectors = ([p for p in itertools.product([0, 1, -1], repeat=3)])
    full_merge_dict = defaultdict(set)
    merging_ccs = defaultdict(set)

    for slice_locations in shared_slices:
        # print(slice_locations)
        left_side = slice_locations - 1
        right_side = slice_locations

        # Iterate through Left Conn Components that have nodes on the boundary
        # Find nodes on the other side and their corresponding CC label indicating a need to merge

        for cc_label in nodes_to_left_of_boundary[left_side].keys():
            # print(cc_label)

            cc_coords_to_check = nodes_to_left_of_boundary[left_side][cc_label]
            for left_node in cc_coords_to_check:
                for vect in movement_vectors:
                    node_to_check_on_other_side = tuple(list(map(add, tuple(left_node), vect)))
                    if node_to_check_on_other_side in nodes_to_right_of_boundary[right_side]:
                        right_cc = nodes_to_right_of_boundary[right_side][node_to_check_on_other_side]

                        # Update Neighbors Dictionary
                        # print('IM ADDING {} to {} Neighbor Dict'.format(node_to_check_on_other_side,left_node))
                        full_neighbors_dict[cc_label][left_node].add(node_to_check_on_other_side)
                        full_neighbors_dict[right_cc][node_to_check_on_other_side].add(left_node)

                        merging_ccs[cc_label].add(right_cc)
                        # print(merging_ccs)

    full_merge_dict = consolidate_conn_components(merging_ccs)

    ################################################################################################################
    # Consolidate Connected Component Labels Across Chunk Slices. E.g. in our image connected components that are
    # actually the same component will have different numerical values if they were impacted by chunking. This
    # will consolidate them
    ################################################################################################################

    # merging these values in full neighbors dict
    for keeping_cc, merging_cc in full_merge_dict.items():
        for merge_cc in merging_cc:
            # pdate full neighbors dict
            full_neighbors_dict[keeping_cc].update(full_neighbors_dict[merge_cc])

            del full_neighbors_dict[merge_cc]

    ################################################################################################################
    # Before building an swc file we need to find the soma location. We can use our ch_1 of segmentation to find
    # the soma centroid if data is not available in lims.
    ################################################################################################################
    ch1_mip_pth = os.path.join(specimen_dir, "MAX_Segmentation_ch1.tif")
    ch1_yz_mip_pth = os.path.join(specimen_dir, "MAX_yz_Segmentation_ch1.tif")

    centroid, connection_threshold, mean_radius, no_soma = get_soma_xyz(ch1_mip_pth, ch1_yz_mip_pth, sp_id)

    ################################################################################################################
    # For each connected component floating around our image stack, we will give it directionality. We will assume
    # the leaf node closest to the soma will be the root of every independent connected component.
    ################################################################################################################

    parent_dict = {}
    parent_dict[centroid] = -1

    for conn_comp in full_neighbors_dict.keys():
        # print('at conn_comp {}'.format(conn_comp))
        neighbors_dict = full_neighbors_dict[conn_comp]
        if len(full_neighbors_dict[conn_comp]) > 2:
            leaf_nodes = [x for x in neighbors_dict.keys() if len(neighbors_dict[x]) == 1]

            # There is no leaf node to start at (loop present) so we will make it by removing a connection in this
            # component that is closest to soma.
            if leaf_nodes == []:
                # find node closest to soma
                dist_dict = {}
                for coord in full_neighbors_dict[conn_comp].keys():
                    dist_to_soma = distance.euclidean(centroid, coord)
                    dist_dict[coord] = dist_to_soma
                start_node = min(dist_dict, key=dist_dict.get)
                while len(full_neighbors_dict[conn_comp][start_node]) > 1:
                    removed = full_neighbors_dict[conn_comp][start_node].pop()
                    full_neighbors_dict[conn_comp][removed].discard(start_node)

                # Check how far it is from soma centroid
                dist = distance.euclidean(centroid, start_node)

                if dist < connection_threshold:
                    start_parent = centroid
                else:
                    start_parent = 0

                assign_parent_child_relation(start_node, start_parent, parent_dict, neighbors_dict)

            # At least one leaf node exists
            else:
                dist_dict = {}
                for coord in leaf_nodes:
                    dist_to_soma = distance.euclidean(centroid, coord)
                    dist_dict[coord] = dist_to_soma
                start_node = min(dist_dict, key=dist_dict.get)

                # Check how far it is from soma cloud
                dist = distance.euclidean(centroid, start_node)

                if dist < connection_threshold:
                    print('assigning soma centroid as the start node')
                    start_parent = centroid
                else:
                    start_parent = 0

                assign_parent_child_relation(start_node, start_parent, parent_dict, neighbors_dict)

    # In case with fake centroid remove centroid from parent dict and centroid
    if no_soma == True:
        parent_dict.pop(centroid)
        for k, v in parent_dict.items():
            if v == centroid:
                parent_dict[k] = -1

    # number each node for swc format
    ct = 0
    big_node_dict = {}
    for j in parent_dict.keys():
        ct += 1
        big_node_dict[tuple(j)] = ct

    # Load node type labels
    skeleton_labeled = pd.read_csv(skeleton_labels_file)
    skeleton_coord_labels_dict = {}
    for n in skeleton_labeled.index:
        skeleton_coord_labels_dict[
            (skeleton_labeled.loc[n].values[0], skeleton_labeled.loc[n].values[1], skeleton_labeled.loc[n].values[2])] = \
            skeleton_labeled.loc[n].values[3]

    # Make swc list for swc file writing
    swc_list = []
    for k, v in parent_dict.items():
        # id,type,x,y,z,r,pid
        if v == 0:
            parent = -1
            node_type = skeleton_coord_labels_dict[k]
            radius = 1
        elif v == -1:
            parent = -1
            node_type = 1
            radius = mean_radius
        else:
            parent = big_node_dict[v]
            node_type = skeleton_coord_labels_dict[k]
            radius = 1

        swc_line = [big_node_dict[k]] + [node_type] + list(k) + [radius] + [parent]

        swc_list.append(swc_line)

    # Organize Outputs
    swc_outdir = os.path.join(specimen_dir, "SWC")
    raw_swc_outdir = os.path.join(swc_outdir, "Raw")
    autotrace_byproducts = os.path.join(specimen_dir, 'Byproducts')

    for chk_dir in [swc_outdir, raw_swc_outdir, autotrace_byproducts]:
        if not os.path.exists(chk_dir):
            os.mkdir(chk_dir)

    # Write swc file
    swc_path = os.path.join(raw_swc_outdir, '{}_{}_{}_1.0.swc'.format(sp_id, model_and_version, autotrace_code_version))
    with open(swc_path, 'w') as f:
        f.write('# id,type,x,y,z,r,pid')
        f.write('\n')
        for sublist in swc_list:
            for val in sublist:
                f.write(str(val))
                f.write(' ')
            f.write('\n')
    print('finished writing swc for specimen {}'.format(sp_id))

    # Move Byproducts
    byproduct_files = [f for f in os.listdir(specimen_dir) if os.path.isfile(os.path.join(specimen_dir, f))]
    for bypro_file in byproduct_files:
        src = os.path.join(specimen_dir, bypro_file)
        dst = os.path.join(autotrace_byproducts, bypro_file)
        shutil.move(src, dst)

    # Clean Up
    directories_to_remove = ["Chunks_of_32", "Chunks_of_32_Left", "Chunks_of_32_Right",
                             "Segmentation", "Left_Segmentation", "Right_Segmentation",
                             "Skeleton", "Left_Skeleton", "Right_Skeleton",
                             "Single_Tif_Images", "Single_Tif_Images_Left", "Single_Tif_Images_Right"]
    print("Cleaning Up:")
    for dir_name in directories_to_remove:
        full_dir_name = os.path.join(specimen_dir, dir_name)
        if os.path.exists(full_dir_name):
            print(full_dir_name)
            shutil.rmtree(full_dir_name)
