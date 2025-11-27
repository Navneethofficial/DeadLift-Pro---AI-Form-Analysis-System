"""
Pose Analyzer Module
Handles pose detection, form analysis, and feedback generation for deadlift analysis
"""

import cv2
import numpy as np
import math
import os
from sklearn.linear_model import Ridge

class PoseAnalyzer:
    """
    Main class for analyzing deadlift form using pose estimation
    """
    
    def __init__(self):
        """Initialize the pose analyzer with model and configuration"""
        # Get paths relative to current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Model configuration - look in Model Files folder
        self.MODE = "MPI"
        
        # Try multiple possible locations for model files
        possible_paths = [
            # In Model Files folder at project root
            (os.path.join(project_root, 'Model Files', 'pose', 'mpi', 'pose_deploy_linevec_faster_4_stages.prototxt'),
             os.path.join(project_root, 'Model Files', 'pose', 'mpi', 'pose_iter_160000.caffemodel')),
            # In pose folder at project root
            (os.path.join(project_root, 'pose', 'mpi', 'pose_deploy_linevec_faster_4_stages.prototxt'),
             os.path.join(project_root, 'pose', 'mpi', 'pose_iter_160000.caffemodel')),
            # In Backend Files/pose folder
            (os.path.join(current_dir, 'pose', 'mpi', 'pose_deploy_linevec_faster_4_stages.prototxt'),
             os.path.join(current_dir, 'pose', 'mpi', 'pose_iter_160000.caffemodel')),
        ]
        
        # Find which path exists
        self.protoFile = None
        self.weightsFile = None
        
        for proto, weights in possible_paths:
            if os.path.exists(proto) and os.path.exists(weights):
                self.protoFile = proto
                self.weightsFile = weights
                print(f"✓ Found model files at: {os.path.dirname(proto)}")
                break
        
        if self.protoFile is None:
            print("⚠️  Warning: Model files not found in any of these locations:")
            for proto, weights in possible_paths:
                print(f"   - {os.path.dirname(proto)}")
            print("\nPlease download OpenPose MPI model files:")
            print("  1. pose_deploy_linevec_faster_4_stages.prototxt")
            print("  2. pose_iter_160000.caffemodel")
            print("\nAnd place them in one of the above locations.")
        
        self.nPoints = 15
        
        # Define skeleton connections for MPI model
        self.POSE_PAIRS = [
            [0,1], [1,2], [2,3], [3,4],      # Head to right arm
            [1,5], [5,6], [6,7],              # Neck to left arm
            [1,14],                            # Neck to chest
            [-1,8], [8,9], [9,10],            # Right hip to ankle
            [-1,11], [11,12], [12,13],        # Left hip to ankle
            [14,-1]                            # Chest to center
        ]
        
        # Load the neural network
        self.net = None
        if self.protoFile and self.weightsFile:
            try:
                self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
                print("✓ Pose detection model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading pose model: {e}")
                self.net = None
        
        # Detection parameters
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1
        
        # Initialize classifier
        self.clf = self._initialize_classifier(project_root)
        
        # Joint names for better feedback
        self.joint_names = [
            "Head", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
            "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip",
            "Right Knee", "Right Ankle", "Left Hip", "Left Knee", 
            "Left Ankle", "Chest"
        ]
        
        # Form thresholds
        self.BACK_ANGLE_MIN = 160
        self.BACK_ANGLE_MAX = 200
        self.HIP_ANGLE_MIN = 30
        self.HIP_ANGLE_MAX = 90
        self.KNEE_ANGLE_MIN = 160
        self.KNEE_ANGLE_MAX = 180
        
    def _initialize_classifier(self, project_root):
        """Initialize the classifier with pre-trained data or default model"""
        clf = Ridge(alpha=1.0)
        
        # Try to load pre-trained model if available
        model_paths = [
            os.path.join(project_root, 'trained_model_ridge.pkl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model_ridge.pkl')
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        clf = pickle.load(f)
                    print(f"✓ Loaded pre-trained model from {model_path}")
                    return clf
                except Exception as e:
                    print(f"⚠ Could not load pre-trained model: {e}")
        
        # Load training data from Training Data folder if available
        x_train_good, x_train_bad = self._load_training_data(project_root)
        
        if len(x_train_good) > 0 and len(x_train_bad) > 0:
            try:
                clf.fit(x_train_good + x_train_bad, 
                       [1] * len(x_train_good) + [0] * len(x_train_bad))
                print(f"✓ Trained classifier with {len(x_train_good)} good and {len(x_train_bad)} bad samples")
            except Exception as e:
                print(f"⚠ Could not train classifier: {e}")
        else:
            print("ℹ No training data found. Using heuristic-based scoring.")
        
        return clf
    
    def _load_training_data(self, project_root):
        """Load training data from Training Data folder"""
        x_train_good = []
        x_train_bad = []
        
        # Try multiple locations for training data
        training_locations = [
            os.path.join(project_root, 'Training Data'),
            os.path.join(project_root, 'good'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'good')
        ]
        
        # Load good postures
        for base_path in training_locations:
            good_folder = os.path.join(base_path, 'good') if 'Training Data' in base_path else base_path
            if os.path.exists(good_folder):
                for filename in os.listdir(good_folder):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            filepath = os.path.join(good_folder, filename)
                            frame = cv2.imread(filepath)
                            if frame is not None:
                                points = self.detect_keypoints(frame)
                                features = self._extract_features(points)
                                if features:
                                    x_train_good.append(features)
                        except Exception as e:
                            pass
                if x_train_good:
                    print(f"✓ Loaded {len(x_train_good)} good posture samples from {good_folder}")
                    break
        
        # Load bad postures
        for base_path in training_locations:
            bad_folder = os.path.join(base_path, 'bad') if 'Training Data' in base_path or base_path.endswith('good') else os.path.join(os.path.dirname(base_path), 'bad')
            if os.path.exists(bad_folder):
                for filename in os.listdir(bad_folder):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            filepath = os.path.join(bad_folder, filename)
                            frame = cv2.imread(filepath)
                            if frame is not None:
                                points = self.detect_keypoints(frame)
                                features = self._extract_features(points)
                                if features:
                                    x_train_bad.append(features)
                        except Exception as e:
                            pass
                if x_train_bad:
                    print(f"✓ Loaded {len(x_train_bad)} bad posture samples from {bad_folder}")
                    break
        
        return x_train_good, x_train_bad
    
    def _extract_features(self, points):
        """Extract feature vector from keypoints"""
        features = []
        for point in points:
            if point:
                features.append(point[0])
                features.append(point[1])
            else:
                features.append(-1)
                features.append(-1)
        return features if len(features) == 30 else None
    
    def findSquaredDist(self, A, B):
        """Calculate squared Euclidean distance between two points"""
        if A is None or B is None:
            return 0
        return (A[0] - B[0])**2 + (A[1] - B[1])**2
    
    def findAngle(self, A, B, C):
        """
        Calculate angle at point B formed by points A, B, C using law of cosines
        Returns angle in degrees
        """
        if A is None or B is None or C is None:
            return None
        
        # Create vectors
        BA = np.array([A[0] - B[0], A[1] - B[1]])
        BC = np.array([C[0] - B[0], C[1] - B[1]])
        
        # Calculate angle using dot product
        cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def findAngleFromSpine(self, point, A, B):
        """
        Calculate angle from spinal baseline
        Used for checking back neutrality
        """
        if point is None or A is None or B is None:
            return None
        
        a_2 = self.findSquaredDist(A, point)
        b_2 = self.findSquaredDist(point, B)
        c_2 = self.findSquaredDist(A, B)
        
        # Law of cosines
        cos_A = (b_2 + c_2 - a_2) / (2 * (b_2 * c_2)**0.5 + 1e-6)
        angle_A = math.acos(np.clip(cos_A, -1, 1)) * 180 / math.pi
        
        cos_B = (a_2 + c_2 - b_2) / (2 * (a_2 * c_2)**0.5 + 1e-6)
        angle_B = math.acos(np.clip(cos_B, -1, 1)) * 180 / math.pi
        
        return max(angle_A, angle_B)
    
    def sigmoid(self, x):
        """Sigmoid activation function for score normalization"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def detect_keypoints(self, frame):
        """
        Detect pose keypoints in the frame using OpenPose
        Returns list of (x, y) tuples or None for undetected joints
        """
        if self.net is None:
            print("⚠ Pose detection model not loaded")
            return [None] * self.nPoints
        
        frameHeight, frameWidth = frame.shape[:2]
        
        # Prepare input blob
        inpBlob = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (self.inWidth, self.inHeight),
            (0, 0, 0), swapRB=False, crop=False
        )
        
        # Forward pass through network
        self.net.setInput(inpBlob)
        output = self.net.forward()
        
        H = output.shape[2]
        W = output.shape[3]
        
        # Extract keypoints
        points = []
        for i in range(self.nPoints):
            # Get confidence map for this body part
            probMap = output[0, i, :, :]
            
            # Find global maximum
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale point to original image size
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            
            # Add point if confidence exceeds threshold
            if prob > self.threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)
        
        return points
    
    def calculate_metrics(self, points):
        """
        Calculate form metrics from detected keypoints
        Returns dictionary of metrics
        """
        metrics = {}
        
        # Back angle (spine neutrality) - Critical for deadlift
        if points[1] and points[8] and points[14]:  # Neck, Hip, Chest
            neck = points[1]
            hip = points[8] if points[8] else points[11]  # Use right or left hip
            chest = points[14]
            
            # Calculate angle at chest point
            back_angle = self.findAngle(neck, chest, hip)
            if back_angle is not None:
                metrics['back_angle'] = back_angle
                
                # Evaluate back form
                if self.BACK_ANGLE_MIN <= back_angle <= self.BACK_ANGLE_MAX:
                    metrics['back_form'] = 'Good'
                else:
                    metrics['back_form'] = 'Needs Improvement'
        
        # Hip hinge angle - Shows proper hip drive
        if points[1] and points[8] and points[9]:  # Neck, Hip, Knee
            hip_angle = self.findAngle(points[1], points[8], points[9])
            if hip_angle is not None:
                metrics['hip_angle'] = hip_angle
                
                if self.HIP_ANGLE_MIN <= hip_angle <= self.HIP_ANGLE_MAX:
                    metrics['hip_hinge'] = 'Good'
                else:
                    metrics['hip_hinge'] = 'Needs Improvement'
        
        # Also check left hip if right is not available
        if 'hip_angle' not in metrics and points[1] and points[11] and points[12]:
            hip_angle = self.findAngle(points[1], points[11], points[12])
            if hip_angle is not None:
                metrics['hip_angle'] = hip_angle
                
                if self.HIP_ANGLE_MIN <= hip_angle <= self.HIP_ANGLE_MAX:
                    metrics['hip_hinge'] = 'Good'
                else:
                    metrics['hip_hinge'] = 'Needs Improvement'
        
        # Knee angle - Should be slightly bent
        if points[8] and points[9] and points[10]:  # Hip, Knee, Ankle
            knee_angle = self.findAngle(points[8], points[9], points[10])
            if knee_angle is not None:
                metrics['knee_angle'] = knee_angle
                
                if self.KNEE_ANGLE_MIN <= knee_angle <= self.KNEE_ANGLE_MAX:
                    metrics['knee_form'] = 'Good'
                else:
                    metrics['knee_form'] = 'Slightly Bent (Check)'
        
        # Also check left knee
        if 'knee_angle' not in metrics and points[11] and points[12] and points[13]:
            knee_angle = self.findAngle(points[11], points[12], points[13])
            if knee_angle is not None:
                metrics['knee_angle'] = knee_angle
                
                if self.KNEE_ANGLE_MIN <= knee_angle <= self.KNEE_ANGLE_MAX:
                    metrics['knee_form'] = 'Good'
                else:
                    metrics['knee_form'] = 'Slightly Bent (Check)'
        
        # Bar path (wrist to ankle alignment) - Bar should be over midfoot
        if points[4] and points[7] and points[10] and points[13]:  # Both wrists and ankles
            avg_wrist_x = (points[4][0] + points[7][0]) / 2
            avg_ankle_x = (points[10][0] + points[13][0]) / 2
            bar_distance = abs(avg_wrist_x - avg_ankle_x)
            
            # Estimate reference distance (thigh length)
            if points[8] and points[9]:
                thigh_length = (self.findSquaredDist(points[8], points[9]))**0.5
                metrics['bar_distance'] = bar_distance
                metrics['bar_distance_ratio'] = bar_distance / (thigh_length + 1e-6)
                
                # Bar should be close to body (within 30% of thigh length)
                if bar_distance < thigh_length * 0.3:
                    metrics['bar_path'] = 'Good'
                else:
                    metrics['bar_path'] = 'Too Far Forward'
        
        # Shoulder alignment - Check if shoulders are over bar
        if points[2] and points[5] and points[4] and points[7]:  # Shoulders and wrists
            avg_shoulder_x = (points[2][0] + points[5][0]) / 2
            avg_wrist_x = (points[4][0] + points[7][0]) / 2
            shoulder_alignment = abs(avg_shoulder_x - avg_wrist_x)
            
            metrics['shoulder_alignment'] = shoulder_alignment
            
            # Shoulders should be slightly ahead of bar
            if points[8]:  # Use hip as reference
                torso_length = (self.findSquaredDist(points[1], points[8]))**0.5
                if shoulder_alignment < torso_length * 0.15:
                    metrics['shoulder_position'] = 'Good'
                else:
                    metrics['shoulder_position'] = 'Check Position'
        
        return metrics
    
    def generate_feedback(self, metrics, score):
        """
        Generate detailed feedback based on metrics and score
        Returns list of feedback strings
        """
        feedback = []
        
        # Overall assessment
        if score >= 0.8:
            feedback.append("🏆 Excellent form! Keep maintaining this technique!")
        elif score >= 0.6:
            feedback.append("✓ Good form overall with minor improvements needed")
        else:
            feedback.append("⚠ Form needs attention - review the points below carefully")
        
        # Back form feedback
        if 'back_form' in metrics:
            if metrics['back_form'] == 'Needs Improvement':
                angle = metrics.get('back_angle', 0)
                if angle < self.BACK_ANGLE_MIN:
                    feedback.append(f"• Back: Your back is rounded ({angle:.1f}°). Focus on neutral spine - chest up, shoulders back")
                else:
                    feedback.append(f"• Back: Over-extended ({angle:.1f}°). Maintain natural spine curve")
            else:
                feedback.append(f"✓ Back: Excellent neutral spine position ({metrics.get('back_angle', 0):.1f}°)")
        
        # Hip hinge feedback
        if 'hip_hinge' in metrics:
            if metrics['hip_hinge'] == 'Needs Improvement':
                angle = metrics.get('hip_angle', 0)
                if angle < self.HIP_ANGLE_MIN:
                    feedback.append(f"• Hips: Not enough hip hinge ({angle:.1f}°). Push hips back more at start")
                else:
                    feedback.append(f"• Hips: Too much bend ({angle:.1f}°). Drive through hips more")
            else:
                feedback.append(f"✓ Hips: Good hip hinge mechanics ({metrics.get('hip_angle', 0):.1f}°)")
        
        # Knee feedback
        if 'knee_form' in metrics and metrics['knee_form'] != 'Good':
            angle = metrics.get('knee_angle', 0)
            if angle < self.KNEE_ANGLE_MIN:
                feedback.append(f"• Knees: Too much knee bend ({angle:.1f}°). This becomes more of a squat")
            else:
                feedback.append(f"• Knees: Keep slight bend, avoid locking out ({angle:.1f}°)")
        elif 'knee_form' in metrics:
            feedback.append(f"✓ Knees: Good position with slight bend ({metrics.get('knee_angle', 0):.1f}°)")
        
        # Bar path feedback
        if 'bar_path' in metrics and metrics['bar_path'] != 'Good':
            feedback.append("• Bar Path: Keep the bar closer to your body throughout the lift")
            feedback.append("  Tip: Bar should travel in a straight vertical line over midfoot")
        elif 'bar_path' in metrics:
            feedback.append("✓ Bar Path: Excellent - bar stays close to body")
        
        # Shoulder position feedback
        if 'shoulder_position' in metrics and metrics['shoulder_position'] != 'Good':
            feedback.append("• Shoulders: Ensure shoulders are slightly ahead of the bar at start")
        
        # General tips based on score
        if score < 0.6:
            feedback.append("\n💡 Tips for improvement:")
            feedback.append("  1. Record from the side for best analysis")
            feedback.append("  2. Start with lighter weight to perfect form")
            feedback.append("  3. Focus on one cue at a time")
            feedback.append("  4. Consider working with a coach")
        
        return feedback
    
    def draw_annotations(self, frame, points, metrics, score):
        """
        Draw skeleton and annotations on the frame with side panel layout
        Returns annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Determine color based on score
        if score >= 0.7:
            skeleton_color = (0, 255, 0)      # Green - Good
            rating = "EXCELLENT"
        elif score >= 0.5:
            skeleton_color = (0, 165, 255)    # Orange - Moderate
            rating = "GOOD"
        else:
            skeleton_color = (0, 0, 255)      # Red - Poor
            rating = "NEEDS WORK"
        
        # Draw skeleton lines
        for pair in self.POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            
            if partA >= 0 and partB >= 0 and partA < len(points) and partB < len(points):
                if points[partA] and points[partB]:
                    cv2.line(annotated, points[partA], points[partB], 
                            skeleton_color, 3, lineType=cv2.LINE_AA)
        
        # Draw keypoints
        for i, point in enumerate(points):
            if point:
                cv2.circle(annotated, point, 8, (255, 255, 255), 
                          thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(annotated, point, 6, skeleton_color, 
                          thickness=-1, lineType=cv2.FILLED)
        
        # Create side panel for information
        panel_width = 400
        panel_x = w - panel_width - 20
        panel_y = 20
        panel_height = 280
        
        # Draw semi-transparent background for side panel
        overlay = annotated.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        
        # Draw panel border
        cv2.rectangle(annotated, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     skeleton_color, 3)
        
        # Text starting positions
        text_x = panel_x + 20
        text_y = panel_y + 45
        
        # Title
        cv2.putText(annotated, "FORM ANALYSIS", (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        text_y += 45
        
        # Score with large display
        score_text = f"Score: {score*100:.1f}%"
        cv2.putText(annotated, score_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # Rating next to score
        cv2.putText(annotated, rating, (text_x + 210, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, skeleton_color, 2, lineType=cv2.LINE_AA)
        
        text_y += 50
        
        # Separator line
        cv2.line(annotated, (text_x, text_y - 15), 
                (panel_x + panel_width - 20, text_y - 15), 
                (100, 100, 100), 1)
        
        # Display metrics with proper spacing
        line_height = 32
        font_scale = 0.55
        
        if 'back_form' in metrics:
            status = "+" if metrics['back_form'] == 'Good' else "X"
            color = (0, 255, 0) if metrics['back_form'] == 'Good' else (0, 0, 255)
            
            # Main text
            main_text = f"{status} Back: {metrics['back_form']}"
            cv2.putText(annotated, main_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, lineType=cv2.LINE_AA)
            
            # Angle on same line, right aligned
            if 'back_angle' in metrics:
                angle_text = f"{metrics['back_angle']:.1f} deg"
                cv2.putText(annotated, angle_text, (text_x + 260, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1, lineType=cv2.LINE_AA)
            
            text_y += line_height
        
        if 'hip_hinge' in metrics:
            status = "+" if metrics['hip_hinge'] == 'Good' else "X"
            color = (0, 255, 0) if metrics['hip_hinge'] == 'Good' else (0, 0, 255)
            
            main_text = f"{status} Hip Hinge: {metrics['hip_hinge']}"
            cv2.putText(annotated, main_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, lineType=cv2.LINE_AA)
            
            if 'hip_angle' in metrics:
                angle_text = f"{metrics['hip_angle']:.1f} deg"
                cv2.putText(annotated, angle_text, (text_x + 260, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1, lineType=cv2.LINE_AA)
            
            text_y += line_height
        
        if 'knee_form' in metrics:
            status = "+" if metrics['knee_form'] == 'Good' else "X"
            color = (0, 255, 0) if metrics['knee_form'] == 'Good' else (0, 165, 255)
            
            main_text = f"{status} Knees: {metrics['knee_form']}"
            cv2.putText(annotated, main_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, lineType=cv2.LINE_AA)
            
            if 'knee_angle' in metrics:
                angle_text = f"{metrics['knee_angle']:.1f} deg"
                cv2.putText(annotated, angle_text, (text_x + 260, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1, lineType=cv2.LINE_AA)
            
            text_y += line_height
        
        if 'bar_path' in metrics:
            status = "+" if metrics['bar_path'] == 'Good' else "X"
            color = (0, 255, 0) if metrics['bar_path'] == 'Good' else (0, 0, 255)
            
            main_text = f"{status} Bar Path: {metrics['bar_path']}"
            cv2.putText(annotated, main_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, lineType=cv2.LINE_AA)
            
            text_y += line_height
        
        if 'shoulder_position' in metrics:
            status = "+" if metrics['shoulder_position'] == 'Good' else "X"
            color = (0, 255, 0) if metrics['shoulder_position'] == 'Good' else (0, 165, 255)
            
            main_text = f"{status} Shoulders: {metrics['shoulder_position']}"
            cv2.putText(annotated, main_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, lineType=cv2.LINE_AA)
        
        return annotated
    
    def analyze_image(self, image_path):
        """
        Analyze a single image for deadlift form
        Returns dictionary with score, metrics, feedback, and annotated frame
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect keypoints
        points = self.detect_keypoints(frame)
        
        # Check if any keypoints were detected
        detected_points = [p for p in points if p is not None]
        if len(detected_points) < 5:
            raise ValueError("Could not detect enough keypoints. Ensure full body is visible with good lighting.")
        
        # Calculate metrics
        metrics = self.calculate_metrics(points)
        
        # Prepare feature vector for classification
        features = self._extract_features(points)
        
        # Calculate score
        if features and len(features) == 30:
            try:
                raw_score = self.clf.predict([features])[0]
                score = self.sigmoid((raw_score - 0.5) * 6)
            except:
                # Fallback to heuristic scoring
                score = self._calculate_heuristic_score(metrics)
        else:
            score = self._calculate_heuristic_score(metrics)
        
        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))
        
        # Generate feedback
        feedback = self.generate_feedback(metrics, score)
        
        # Draw annotations
        annotated_frame = self.draw_annotations(frame, points, metrics, score)
        
        return {
            'score': float(score),
            'metrics': metrics,
            'feedback': feedback,
            'keypoints': [(p[0], p[1]) if p else None for p in points],
            'annotated_frame': annotated_frame
        }
    
    def _calculate_heuristic_score(self, metrics):
        """
        Calculate score based on metrics when classifier is not available
        Uses rule-based approach with severity-based penalties
        """
        score = 1.0
        penalties = []
        
        # Back form penalty (most critical for deadlift)
        if 'back_form' in metrics and metrics['back_form'] != 'Good':
            # Check severity based on angle deviation
            if 'back_angle' in metrics:
                angle = metrics['back_angle']
                deviation = max(
                    0,
                    self.BACK_ANGLE_MIN - angle,
                    angle - self.BACK_ANGLE_MAX
                )
                # Penalty increases with deviation (0.15 to 0.35)
                penalty = min(0.35, 0.15 + (deviation / 100))
            else:
                penalty = 0.25
            score -= penalty
            penalties.append(('back', penalty))
        
        # Hip hinge penalty
        if 'hip_hinge' in metrics and metrics['hip_hinge'] != 'Good':
            if 'hip_angle' in metrics:
                angle = metrics['hip_angle']
                deviation = max(
                    0,
                    self.HIP_ANGLE_MIN - angle,
                    angle - self.HIP_ANGLE_MAX
                )
                penalty = min(0.25, 0.1 + (deviation / 150))
            else:
                penalty = 0.15
            score -= penalty
            penalties.append(('hip', penalty))
        
        # Knee form penalty (less critical)
        if 'knee_form' in metrics and metrics['knee_form'] != 'Good':
            penalty = 0.1
            score -= penalty
            penalties.append(('knee', penalty))
        
        # Bar path penalty
        if 'bar_path' in metrics and metrics['bar_path'] != 'Good':
            penalty = 0.15
            score -= penalty
            penalties.append(('bar_path', penalty))
        
        # Shoulder position penalty (minor)
        if 'shoulder_position' in metrics and metrics['shoulder_position'] != 'Good':
            penalty = 0.08
            score -= penalty
            penalties.append(('shoulder', penalty))
        
        # Add small bonus if multiple things are good
        good_count = sum(1 for k, v in metrics.items() if v == 'Good')
        if good_count >= 3:
            score += 0.05
        
        # Ensure score is in valid range (minimum 15% for detection)
        return max(0.15, min(1.0, score))