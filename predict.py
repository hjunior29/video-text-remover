"""
Video Text Remover for Replicate - Video Processing
Detects and removes hardcoded text overlays from videos using YOLO + Inpainting
"""

from cog import BasePredictor, Input, Path
import cv2
import numpy as np
import onnxruntime as ort
from typing import List
import tempfile
import os
import subprocess
import shutil

class Predictor(BasePredictor):
    """Video Text Remover detection and removal predictor for videos"""

    def setup(self):
        """Load ONNX model once when container starts"""
        import time
        start_time = time.time()

        print("=" * 60)
        print("STARTING VIDEO TEXT REMOVER SETUP")
        print("=" * 60)

        # Check model file
        model_path = "models/text_detector/converted_best.onnx"
        print(f"Step 1/3: Checking model file...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   - Model found: {model_path}")
        print(f"   - Size: {model_size:.1f} MB")

        # Configure providers (auto-detect GPU/CPU)
        print(f"\nStep 2/3: Configuring ONNX Runtime...")

        # Optimize session options for faster startup
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.intra_op_num_threads = 2  # Limit threads for faster startup
        sess_options.log_severity_level = 3  # Reduce logging

        # Auto-detect available providers (GPU first, then CPU fallback)
        available_providers = ort.get_available_providers()

        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"   - Provider: CUDA GPU (NVIDIA)")
        elif 'TensorrtExecutionProvider' in available_providers:
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"   - Provider: TensorRT GPU (optimized)")
        else:
            providers = ['CPUExecutionProvider']
            print(f"   - Provider: CPU")

        print(f"   - Threads: 2 (faster initialization)")
        print(f"   - Available providers: {', '.join(available_providers)}")

        # Load model
        print(f"\nStep 3/3: Loading ONNX model...")
        load_start = time.time()

        self.ort_session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=sess_options
        )

        load_time = time.time() - load_start
        print(f"   - Model loaded in {load_time:.2f}s")

        # Get model info
        inputs = self.ort_session.get_inputs()
        outputs = self.ort_session.get_outputs()

        self.input_name = inputs[0].name if inputs else None
        self.output_names = [output.name for output in outputs] if outputs else []

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("SETUP COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Input: {self.input_name}")
        print(f"   Outputs: {', '.join(self.output_names)}")
        print(f"   Ready to process videos")
        print("=" * 60 + "\n")

    def predict(
        self,
        video: Path = Input(
            description="Input video file with hardcoded text to remove. Supports MP4, AVI, MOV, and other common video formats.",
        ),
        method: str = Input(
            description="Video Text Remover removal algorithm. 'hybrid' (recommended): Best quality using context-aware inpainting. 'inpaint': Fast TELEA inpainting. 'inpaint_ns': Navier-Stokes inpainting. 'blur': Gaussian blur. 'black': Fill with black. 'background': Fill with surrounding color.",
            default="hybrid",
            choices=["hybrid", "inpaint", "inpaint_ns", "blur", "black", "background"],
        ),
        conf_threshold: float = Input(
            description="Detection confidence threshold (0.0-1.0). Lower values detect more text but may include false positives. Recommended: 0.25",
            default=0.25,
            ge=0.0,
            le=1.0,
        ),
        iou_threshold: float = Input(
            description="Intersection-over-Union threshold for removing duplicate detections (0.0-1.0). Higher values keep more overlapping boxes. Recommended: 0.45",
            default=0.45,
            ge=0.0,
            le=1.0,
        ),
        margin: int = Input(
            description="Extra pixels to expand around detected text regions (0-20). Higher values ensure complete removal but may remove more content. Recommended: 5",
            default=5,
            ge=0,
            le=20,
        ),
    ) -> Path:
        """Remove hardcoded text from video using AI detection and inpainting

        Returns:
            Path: Path to the processed video file with text removed
        """

        print(f"\nProcessing video: {video}")
        print(f"   - Method: {method}")
        print(f"   - Confidence: {conf_threshold}")
        print(f"   - Margin: {margin}px")

        # Validate input file exists
        if not os.path.exists(str(video)):
            raise ValueError(f"Video file not found: {video}")

        # Open video
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file. Please ensure it's a valid video format (MP4, AVI, MOV, etc.): {video}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate video properties
        if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid video properties. FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")

        print(f"\nVideo info:")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - Total frames: {total_frames}")
        print(f"   - Duration: {total_frames/fps:.2f}s")

        # Create temporary directory for frames
        frames_dir = Path(tempfile.mkdtemp()) / "frames"
        frames_dir.mkdir(exist_ok=True)
        output_path = Path(tempfile.mkdtemp()) / "output.mp4"

        print(f"\nProcessing frames...")

        # Process each frame
        frames_with_text = 0
        total_detections = 0
        frame_num = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in buffer
                    if frame_buffer:
                        for buf_frame, buf_boxes in zip(frame_buffer, boxes_buffer):
                            # Update stats
                            if len(buf_boxes) > 0:
                                frames_with_text += 1
                                total_detections += len(buf_boxes)

                                # Remove text with optimized inpainting
                                t2 = time.time()
                                for box in buf_boxes:
                                    buf_frame = self._remove_text_optimized(buf_frame, box, method, margin)
                                inpainting_time += time.time() - t2

                            # Write frame directly to FFmpeg pipe
                            t3 = time.time()
                            try:
                                ffmpeg_process.stdin.write(buf_frame.tobytes())
                            except BrokenPipeError:
                                raise RuntimeError("FFmpeg pipe broken - encoding failed")
                            io_time += time.time() - t3
                    break

                # Detect text
                boxes = self._detect_onnx(frame, conf_threshold, iou_threshold)

                boxes_buffer.append(boxes)
                frame_num += 1

                # Process batch when buffer is full or at end
                if len(frame_buffer) >= BATCH_SIZE:
                    # Process all frames in buffer
                    for buf_frame, buf_boxes in zip(frame_buffer, boxes_buffer):
                        # Update stats
                        if len(buf_boxes) > 0:
                            frames_with_text += 1
                            total_detections += len(buf_boxes)

                    # Remove text
                    for box in boxes:
                        frame = self._remove_text(frame, box, method, margin)

                # Save frame as image
                frame_path = frames_dir / f"frame_{frame_num:06d}.png"
                cv2.imwrite(str(frame_path), frame)

                    # Progress reporting
                    if frame_num % 30 == 0 or frame_num in [1, 10, 50, 100]:
                        progress = (frame_num / total_frames) * 100
                        print(f"   Progress: {frame_num}/{total_frames} frames ({progress:.1f}%) - Text detected in {frames_with_text} frames")

                    # Clear buffers
                    frame_buffer = []
                    boxes_buffer = []

        except Exception as e:
            # Cleanup on error
            cap.release()
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            raise RuntimeError(f"Error during video processing at frame {frame_num}/{total_frames}: {str(e)}")
        finally:
            # Always release video capture
            cap.release()

        print(f"\nEncoding video with FFmpeg...")

        # Use FFmpeg to encode the video from frames
        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-framerate', str(fps),
                '-i', str(frames_dir / 'frame_%06d.png'),
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'medium',  # Encoding speed
                '-crf', '23',  # Quality (lower = better, 18-28 is good range)
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-movflags', '+faststart',  # Enable fast start for web playback
                str(output_path)
            ]

            subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True
            )

            print(f"   - Video encoded successfully with H.264")

        except subprocess.CalledProcessError as e:
            # Cleanup on error
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            raise RuntimeError(f"FFmpeg encoding failed: {e.stderr}")
        finally:
            # Cleanup frames directory
            if frames_dir.exists():
                shutil.rmtree(frames_dir)

        # Verify output file was created successfully
        if not output_path.exists():
            raise RuntimeError("Output video file was not created")

        output_size = os.path.getsize(output_path)
        if output_size == 0:
            raise RuntimeError("Output video file is empty")

        # Stats
        print(f"\nRESULTS:")
        print(f"   - Frames processed: {total_frames}")
        print(f"   - Frames with text: {frames_with_text} ({100*frames_with_text/total_frames:.1f}%)")
        print(f"   - Total text regions removed: {total_detections}")
        if total_frames > 0:
            print(f"   - Average detections per frame: {total_detections/total_frames:.2f}")
        print(f"\nOutput: {output_path}")
        print(f"   Size: {output_size / 1024 / 1024:.2f} MB")

        return output_path

    def _preprocess_frame(self, frame: np.ndarray, input_size: int = 640):
        """Preprocess single frame for YOLO inference"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = frame.shape[:2]

        # Prepare input for YOLO (640x640 with padding)
        scale = min(input_size / orig_width, input_size / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        resized = cv2.resize(frame_rgb, (new_width, new_height))
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_x = (input_size - new_width) // 2
        pad_y = (input_size - new_height) // 2
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized

        # Normalize and transpose
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)

        return input_tensor, (orig_width, orig_height, scale, pad_x, pad_y)

    def _postprocess_detections(self, predictions, orig_info, conf_threshold: float, iou_threshold: float):
        """Postprocess YOLO predictions to bounding boxes"""
        boxes = []
        orig_width, orig_height, scale, pad_x, pad_y = orig_info

        if predictions is None:
            return boxes

        if hasattr(predictions, 'shape') and len(predictions.shape) == 3:
            predictions = predictions[0]

        # Process detections
        for pred in predictions:
            if pred is not None and len(pred) >= 5:
                conf = pred[4]
                if conf >= conf_threshold:
                    x_center, y_center, width, height = pred[:4]

                    # Convert to original image coordinates
                    x1 = (x_center - width/2 - pad_x) / scale
                    y1 = (y_center - height/2 - pad_y) / scale
                    x2 = (x_center + width/2 - pad_x) / scale
                    y2 = (y_center + height/2 - pad_y) / scale

                    # Clamp to image bounds
                    x1 = max(0, min(x1, orig_width))
                    y1 = max(0, min(y1, orig_height))
                    x2 = max(0, min(x2, orig_width))
                    y2 = max(0, min(y2, orig_height))

                    boxes.append([int(x1), int(y1), int(x2), int(y2)])

        # Apply NMS
        if len(boxes) > 1:
            boxes = self._apply_nms(boxes, iou_threshold)

        return boxes

    def _detect_onnx_batch(
        self,
        frames: List[np.ndarray],
        conf_threshold: float,
        iou_threshold: float
    ) -> List[List[List[int]]]:
        """Detect text overlays in batch of frames using ONNX model"""
        batch_size = len(frames)
        input_size = 640

        try:
            # Preprocess all frames
            input_batch = np.zeros((batch_size, 3, input_size, input_size), dtype=np.float32)
            orig_infos = []

            for i, frame in enumerate(frames):
                input_tensor, orig_info = self._preprocess_frame(frame, input_size)
                input_batch[i] = input_tensor
                orig_infos.append(orig_info)

            # Run batch inference
            outputs = self.ort_session.run(
                self.output_names,
                {self.input_name: input_batch}
            )

            if outputs is None or len(outputs) == 0:
                return [[] for _ in range(batch_size)]

            predictions_batch = outputs[0]

            # Postprocess each frame's predictions
            all_boxes = []
            for i in range(batch_size):
                predictions = predictions_batch[i] if batch_size > 1 else predictions_batch
                boxes = self._postprocess_detections(predictions, orig_infos[i], conf_threshold, iou_threshold)
                all_boxes.append(boxes)

            return all_boxes

        except Exception as e:
            print(f"Batch detection error: {e}")
            return [[] for _ in range(batch_size)]

    def _detect_onnx(
        self,
        frame: np.ndarray,
        conf_threshold: float,
        iou_threshold: float
    ) -> List[List[int]]:
        """Detect text overlays using ONNX model (single frame)"""
        try:
            input_tensor, orig_info = self._preprocess_frame(frame, input_size=640)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Run inference
            outputs = self.ort_session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )

            if outputs is None or len(outputs) == 0:
                return []

            predictions = outputs[0]
            return self._postprocess_detections(predictions, orig_info, conf_threshold, iou_threshold)

        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def _apply_nms(
        self,
        boxes: List[List[int]],
        iou_threshold: float
    ) -> List[List[int]]:
        """Apply Non-Maximum Suppression"""
        if not boxes:
            return []

        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def _remove_text_optimized(
        self,
        frame: np.ndarray,
        box: List[int],
        method: str,
        margin: int
    ) -> np.ndarray:
        """Remove text from frame using optimized method based on box size"""
        x1, y1, x2, y2 = self._expand_box(frame, box, margin)

        # Calculate box area
        box_area = (x2 - x1) * (y2 - y1)
        SMALL_BOX_THRESHOLD = 8000  # pixels (e.g., 100x80)

        # For small boxes, use fast blur instead of slow inpainting
        if box_area < SMALL_BOX_THRESHOLD and method in ["hybrid", "inpaint", "inpaint_ns"]:
            return self._apply_blur(frame, [x1, y1, x2, y2])

        # For larger boxes, use requested method
        if method == "hybrid":
            return self._apply_hybrid(frame, [x1, y1, x2, y2], margin)
        elif method == "inpaint":
            return self._apply_inpaint_telea(frame, [x1, y1, x2, y2])
        elif method == "inpaint_ns":
            return self._apply_inpaint_ns(frame, [x1, y1, x2, y2])
        elif method == "blur":
            return self._apply_blur(frame, [x1, y1, x2, y2])
        elif method == "black":
            frame[y1:y2, x1:x2] = (0, 0, 0)
            return frame
        elif method == "background":
            return self._apply_background(frame, [x1, y1, x2, y2])
        else:
            return self._apply_inpaint_telea(frame, [x1, y1, x2, y2])

    def _remove_text(
        self,
        frame: np.ndarray,
        box: List[int],
        method: str,
        margin: int
    ) -> np.ndarray:
        """Remove text from frame using specified method"""
        x1, y1, x2, y2 = self._expand_box(frame, box, margin)

        if method == "hybrid":
            return self._apply_hybrid(frame, [x1, y1, x2, y2], margin)
        elif method == "inpaint":
            return self._apply_inpaint_telea(frame, [x1, y1, x2, y2])
        elif method == "inpaint_ns":
            return self._apply_inpaint_ns(frame, [x1, y1, x2, y2])
        elif method == "blur":
            return self._apply_blur(frame, [x1, y1, x2, y2])
        elif method == "black":
            frame[y1:y2, x1:x2] = (0, 0, 0)
            return frame
        elif method == "background":
            return self._apply_background(frame, [x1, y1, x2, y2])
        else:
            return self._apply_inpaint_telea(frame, [x1, y1, x2, y2])

    def _expand_box(self, frame, box, margin):
        """Expand bounding box with margin"""
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)

        return [x1, y1, x2, y2]

    def _apply_inpaint_telea(self, frame, box):
        """Apply TELEA inpainting"""
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        result = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return result

    def _apply_inpaint_ns(self, frame, box):
        """Apply Navier-Stokes inpainting"""
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        result = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        return result

    def _apply_blur(self, frame, box):
        """Apply Gaussian blur"""
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            roi_blurred = cv2.GaussianBlur(roi, (51, 51), 30)
            frame[y1:y2, x1:x2] = roi_blurred

        return frame

    def _apply_hybrid(self, frame, box, _margin):
        """Apply hybrid inpainting with expanded context

        Note: margin parameter is not used here as hybrid method
        uses its own fixed context_margin of 20px
        """
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]

        # Expand context (fixed 20px for best quality)
        context_margin = 20
        cx1 = max(0, x1 - context_margin)
        cy1 = max(0, y1 - context_margin)
        cx2 = min(width, x2 + context_margin)
        cy2 = min(height, y2 + context_margin)

        # Extract expanded region
        roi_expanded = frame[cy1:cy2, cx1:cx2].copy()

        # Create local mask
        mask_local = np.zeros(roi_expanded.shape[:2], dtype=np.uint8)
        mask_x1 = x1 - cx1
        mask_y1 = y1 - cy1
        mask_x2 = x2 - cx1
        mask_y2 = y2 - cy1
        mask_local[mask_y1:mask_y2, mask_x1:mask_x2] = 255

        # Inpaint with larger radius
        roi_inpainted = cv2.inpaint(roi_expanded, mask_local, 7, cv2.INPAINT_TELEA)

        # Copy back
        frame[y1:y2, x1:x2] = roi_inpainted[mask_y1:mask_y2, mask_x1:mask_x2]

        return frame

    def _apply_background(self, frame, box):
        """Fill with average background color"""
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]

        # Sample region
        sample_margin = 10
        sx1 = max(0, x1 - sample_margin)
        sy1 = max(0, y1 - sample_margin)
        sx2 = min(width, x2 + sample_margin)
        sy2 = min(height, y2 + sample_margin)

        # Mask for sampling (exclude center)
        mask = np.ones((sy2 - sy1, sx2 - sx1), dtype=np.uint8) * 255
        if sx1 < x1 < sx2 and sy1 < y1 < sy2:
            mask[y1-sy1:y2-sy1, x1-sx1:x2-sx1] = 0

        sample_region = frame[sy1:sy2, sx1:sx2]

        # Calculate mean color
        if mask.any():
            mean_color = cv2.mean(sample_region, mask=mask)[:3]
            frame[y1:y2, x1:x2] = mean_color
        else:
            frame[y1:y2, x1:x2] = (0, 0, 0)

        return frame
