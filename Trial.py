lanes= {('Vehicle moving towards camera', 1): [0, 2], ('Vehicle moving towards camera', 0): [1, 3, 4, 5]}
        for lane_key, vehicle_ids in lanes.items():
            flow_direction, lane = lane_key
            print("Lane:", lane)
            if len(vehicle_ids) > 0:
                reference_vehicle_idx = vehicle_ids[0]  # Take the first vehicle in the lane as reference
                print("Reference vehicle ID:", reference_vehicle_idx)
                # Access the boxes of the reference vehicle
                reference_vehicle_boxes = list(results[reference_vehicle_idx].boxes)
                if reference_vehicle_boxes:  # Check if there are boxes detected for the reference vehicle
                    reference_vehicle_box = reference_vehicle_boxes[0].xyxy[0]
                    reference_centroid = ((reference_vehicle_box[0] + reference_vehicle_box[2]) / 2,
                                          (reference_vehicle_box[1] + reference_vehicle_box[3]) / 2)
                    for followed_vehicle_idx in vehicle_ids[1:]:
                        # Access the boxes of the followed vehicle
                        followed_vehicle_boxes = list(results[followed_vehicle_idx].boxes)
                        if followed_vehicle_boxes:  # Check if there are boxes detected for the followed vehicle
                            followed_vehicle_box = followed_vehicle_boxes[0].xyxy[0]
                            followed_centroid = ((followed_vehicle_box[0] + followed_vehicle_box[2]) / 2,
                                                 (followed_vehicle_box[1] + followed_vehicle_box[3]) / 2)
                            # Calculate distance between centroids
                            distance_pixels = math.sqrt((followed_centroid[0] - reference_centroid[0]) ** 2 +
                                                        (followed_centroid[1] - reference_centroid[1]) ** 2)
                            # Calculate distance in inches
                            known_car_width_inches = 70.0
                            distance_inches = calculate_distance(distance_pixels, known_car_width_inches, focal_length)
                            print(
                                f"Distance between vehicle {reference_vehicle_idx} and vehicle {followed_vehicle_idx}: "
                                f"{distance_inches:.2f} inches")