from utils.video_utils import read_video, save_video
from utils.bbox_utils import measure_distance
from utils.player_stats_drawer_utils import draw_player_stats
from utils.conversions import convert_pixel_distance_to_meters
from tracks.player_track import PlayerTracker
from tracks.ball_track import BallTracker
from court_line_detector import CourtLine_Detector
from mini_court import MiniCourt
import constants
import cv2
from copy import deepcopy
import pandas as pd

import os

def main():
    input_video_path = "input_clips/input_video.mp4"
    frames = read_video(input_video_path)
    if not frames:
        print(f"Error: No frames read from video file {input_video_path}")
        return

    # Provide the correct path to the YOLO model file
    yolo_model_path = "/Users/sunnyg/Desktop/AI_Projects/Tennis_Comp_Vision/yolov8x.pt"

    #Detect Players and Ball
    player_tracker = PlayerTracker(yolo_model_path)
    ball_tracker = BallTracker(model_path="models/last.pt")

    
    # Ensure the directory for the stub path exists
    stub_path = "/Users/sunnyg/Desktop/AI_Projects/Tennis_Comp_Vision/tracker_stubs/playerdetects.pkl"
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)

    player_detections = player_tracker.detect_frames(frames, 
                                                     read_from_stub=True,
                                                     stub_path=stub_path
                                                     )

    ball_detections = ball_tracker.detect_frames(frames, 
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detects.pkl"
                                                     )
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)



    # Court Line Detector 
    court_model_path = "models/keypts_model.pth"
    court_line_detector = CourtLine_Detector(court_model_path)
    court_keypoints = court_line_detector.predict(frames[0])

    # choosing players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt

    mini_court = MiniCourt(frames[0])

    # Detect ball shots

    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)
    

    
     # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)
    player_stats_data = [{
        'frame_num':0,
        'player_6_number_of_shots':0,
        'player_6_total_shot_speed':0,
        'player_6_latest_shot_speed':0,
        'player_6_total_player_speed':0,
        'player_6_latest_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_latest_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_latest_player_speed':0,
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_sec = (end_frame-start_frame)/24 # 24fps

        # distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLES_LINE_WIDTH,
                                                                           mini_court.get_width_mini_court()
                                                                           ) 

    
    # Speed of the ball shot in mph
    speed_of_ball_shot_mph = distance_covered_by_ball_meters / ball_shot_time_in_sec * 2.237
    # player who shot the ball
    player_positions = player_mini_court_detections[start_frame]
    player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))            
    
     # opponent player speed
    opponent_player_id = 2 if player_shot_ball == 6 else 6
    distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
    distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLES_LINE_WIDTH,
                                                                           mini_court.get_width_mini_court()
                                                                           ) 

    speed_of_opponent_mph = distance_covered_by_opponent_meters / ball_shot_time_in_sec * 2.237

    
    current_player_stats= deepcopy(player_stats_data[-1])
    current_player_stats['frame_num'] = start_frame
    current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
    current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot_mph
    current_player_stats[f'player_{player_shot_ball}_latest_shot_speed'] = speed_of_ball_shot_mph

    current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent_mph
    current_player_stats[f'player_{opponent_player_id}_latest_player_speed'] = speed_of_opponent_mph

    player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_6_average_shot_speed'] = player_stats_data_df['player_6_total_shot_speed']/player_stats_data_df['player_6_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_6_average_player_speed'] = player_stats_data_df['player_6_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_6_number_of_shots']


    output_video_frames = player_tracker.draw_bboxes(frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video (output_video_frames, court_keypoints)

    
    
     # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

     # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)   
# Draw frame number on the top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_vids/output_video.avi")

if __name__ == "__main__":
    main()