import cv2
import numpy as np
from pose_analyzer import PoseAnalyzer

class VideoProcessor:
    def __init__(self):
        self.analyzer = PoseAnalyzer()
    
    def process_video(self, input_path, output_path):
        """Process video file and analyze deadlift form"""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_scores = []
        frame_count = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect keypoints
            points = self.analyzer.detect_keypoints(frame)
            
            # Calculate metrics
            metrics = self.analyzer.calculate_metrics(points)
            
            # Prepare data for classification
            temp = []
            for point in points:
                if point:
                    temp.append(point[0])
                    temp.append(point[1])
                else:
                    temp.append(-1)
                    temp.append(-1)
            
            # Calculate score
            try:
                raw_score = self.analyzer.clf.predict([temp])[0]
                score = self.analyzer.sigmoid((raw_score - 0.5) * 6)
            except:
                score = self.analyzer._calculate_heuristic_score(metrics)
            
            frame_scores.append(float(score))
            
            # Draw annotations
            annotated_frame = self.analyzer.draw_annotations(frame, points, metrics, score)
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                       (width - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            
            # Write frame
            out.write(annotated_frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        # Release resources
        cap.release()
        out.release()
        
        # Calculate average score
        avg_score = np.mean(frame_scores) if frame_scores else 0.0
        
        # Generate overall feedback
        feedback = self._generate_video_feedback(frame_scores, avg_score)
        
        print(f"Video processing complete. Average score: {avg_score:.2f}")
        
        return {
            'frame_scores': frame_scores,
            'avg_score': float(avg_score),
            'feedback': feedback,
            'frame_count': frame_count
        }
    
    def _generate_video_feedback(self, frame_scores, avg_score):
        """Generate feedback for entire video"""
        feedback = []
        
        if avg_score >= 0.8:
            feedback.append("🏆 Excellent overall form throughout the lift!")
        elif avg_score >= 0.6:
            feedback.append("✓ Good form with some inconsistencies")
        else:
            feedback.append("⚠ Form needs significant improvement")
        
        # Analyze consistency
        if len(frame_scores) > 0:
            score_std = np.std(frame_scores)
            if score_std < 0.1:
                feedback.append("✓ Consistent form throughout the movement")
            elif score_std < 0.2:
                feedback.append("• Form varies slightly - work on consistency")
            else:
                feedback.append("• Form varies significantly - focus on maintaining position")
        
        # Identify problem phases
        if len(frame_scores) >= 3:
            third = len(frame_scores) // 3
            start_phase = np.mean(frame_scores[:third])
            mid_phase = np.mean(frame_scores[third:2*third])
            end_phase = np.mean(frame_scores[2*third:])
            
            if start_phase < 0.6:
                feedback.append("• Setup phase needs work - focus on starting position")
            if mid_phase < 0.6:
                feedback.append("• Mid-lift form breaks down - maintain tension")
            if end_phase < 0.6:
                feedback.append("• Lockout phase needs attention - control the descent")
        
        return feedback