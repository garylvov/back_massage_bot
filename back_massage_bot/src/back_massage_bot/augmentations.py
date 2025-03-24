import numpy as np
import cv2
from scipy import ndimage


# Define YOLO class mapping with legs grouped together
YOLO_CLASS_MAPPING = [
    "Torso",
    "Head",
    "left_upper_arm",
    "right_upper_arm",
    "left_lower_arm",
    "right_lower_arm",
    "legs"  # All legs (left and right) grouped together
]

# Enhanced Command Pattern for Image Operations
class ImageCommand:
    """Base class for image processing commands"""
    def __init__(self, probability=1.0):
        """
        Initialize the command with a probability of execution
        
        Args:
            probability: Float between 0 and 1 indicating the chance this command will execute
        """
        self.probability = probability
    
    def execute(self, image):
        """Execute the command on the image if probability check passes"""
        if np.random.random() < self.probability:
            return self._apply(image)
        return image
    
    def _apply(self, image):
        """Internal method to be implemented by subclasses"""
        pass
    
    def get_name(self):
        """Get the name of the command for display/labels"""
        return "Generic Command"
    
    def _sample_param(self, param):
        """Sample a parameter value from a range if provided as a tuple"""
        if isinstance(param, tuple) and len(param) == 2:
            return np.random.uniform(param[0], param[1])
        return param

class BlurCommand(ImageCommand):
    def __init__(self, sigma=1.0, probability=1.0):
        super().__init__(probability)
        self.sigma = sigma
    
    def _apply(self, image):
        # Sample sigma from range if provided
        actual_sigma = self._sample_param(self.sigma)
        return ndimage.gaussian_filter(image.astype(float), sigma=actual_sigma)
    
    def get_name(self):
        if isinstance(self.sigma, tuple):
            return f"Blur (σ={self.sigma[0]}-{self.sigma[1]})"
        return f"Blur (σ={self.sigma})"

class ThresholdCommand(ImageCommand):
    def __init__(self, threshold_value=0.5, probability=1.0):
        super().__init__(probability)
        self.threshold_value = threshold_value
    
    def _apply(self, image):
        # Sample threshold from range if provided
        actual_threshold = self._sample_param(self.threshold_value)
        return (image > actual_threshold).astype(float)
    
    def get_name(self):
        if isinstance(self.threshold_value, tuple):
            return f"Threshold ({int(self.threshold_value[0]*255)}-{int(self.threshold_value[1]*255)}/255)"
        return f"Threshold ({int(self.threshold_value*255)}/255)"

class ErosionCommand(ImageCommand):
    def __init__(self, kernel_size=3, iterations=1, probability=1.0):
        super().__init__(probability)
        self.kernel_size = kernel_size
        self.iterations = iterations
    
    def _apply(self, image):
        # Sample parameters from ranges if provided
        actual_kernel_size = int(self._sample_param(self.kernel_size))
        actual_iterations = int(self._sample_param(self.iterations))
        
        kernel = np.ones((actual_kernel_size, actual_kernel_size), np.uint8)
        # Convert to appropriate format for cv2.erode
        img_normalized = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        eroded = cv2.erode(img_normalized, kernel, iterations=actual_iterations)
        # Return to original range
        return eroded.astype(float) / 255.0 if image.max() <= 1.0 else eroded.astype(float)
    
    def get_name(self):
        k_str = f"{self.kernel_size}" if not isinstance(self.kernel_size, tuple) else f"{self.kernel_size[0]}-{self.kernel_size[1]}"
        i_str = f"{self.iterations}" if not isinstance(self.iterations, tuple) else f"{self.iterations[0]}-{self.iterations[1]}"
        return f"Erode (k={k_str}, i={i_str})"

class DilationCommand(ImageCommand):
    def __init__(self, kernel_size=3, iterations=1, shape="rect", probability=1.0):
        super().__init__(probability)
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.shape = shape
    
    def _apply(self, image):
        # Sample parameters from ranges if provided
        actual_kernel_size = int(self._sample_param(self.kernel_size))
        actual_iterations = int(self._sample_param(self.iterations))
        
        # Create kernel based on shape
        if self.shape == "rect":
            kernel = np.ones((actual_kernel_size, actual_kernel_size), np.uint8)
        elif self.shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (actual_kernel_size, actual_kernel_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, 
                                             (actual_kernel_size, actual_kernel_size))
        
        # Handle normalization
        if np.max(image) <= 1.0:
            img_normalized = (image * 255).astype(np.uint8)
            dilated = cv2.dilate(img_normalized, kernel, iterations=actual_iterations)
            return dilated.astype(float) / 255.0
        else:
            return cv2.dilate(image.astype(np.uint8), kernel, 
                             iterations=actual_iterations).astype(float)
    
    def get_name(self):
        k_str = f"{self.kernel_size}" if not isinstance(self.kernel_size, tuple) else f"{self.kernel_size[0]}-{self.kernel_size[1]}"
        i_str = f"{self.iterations}" if not isinstance(self.iterations, tuple) else f"{self.iterations[0]}-{self.iterations[1]}"
        return f"Dilate (k={k_str}, i={i_str}, {self.shape})"

class ClusteredNoiseCommand(ImageCommand):
    """Adds clusters of noise with more control over size and distribution"""
    def __init__(self, 
                 density=(0.05, 0.15),  # Higher density
                 cluster_size=(3, 10),  # Larger clusters
                 value=0.0,            # 0=remove pixels, 1=add pixels
                 target_value=None,    # Value of pixels to target (0=black areas, 1=white areas, None=all)
                 border_focus=0.0,     # Focus clusters near borders (0=uniform, 1=all at borders)
                 probability=1.0):
        super().__init__(probability)
        self.density = density
        self.cluster_size = cluster_size
        self.value = value
        self.target_value = target_value
        self.border_focus = border_focus
    
    def _apply(self, image):
        # Sample parameters
        actual_density = self._sample_param(self.density)
        max_cluster_size = int(self._sample_param(self.cluster_size))
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # Create target mask based on target_value
        if self.target_value is not None:
            if self.target_value == 1.0:
                target_mask = (image > 0.5)  # Target white pixels
            else:
                target_mask = (image < 0.5)  # Target black pixels
        else:
            target_mask = np.ones((h, w), dtype=bool)  # Target all pixels
        
        # Calculate number of clusters based on density and average cluster size
        avg_cluster_area = np.pi * (max_cluster_size/2)**2
        target_pixels = np.sum(target_mask)
        if target_pixels == 0:  # No pixels to target
            return result
            
        num_clusters = int(target_pixels * actual_density / avg_cluster_area)
        
        # Get coordinates of targetable pixels
        target_y, target_x = np.where(target_mask)
        if len(target_y) == 0:
            return result
        
        # Create clusters
        for _ in range(num_clusters):
            # Determine cluster center with optional border focus
            if np.random.random() < self.border_focus:
                # Place near border
                border_type = np.random.randint(0, 4)  # 0=top, 1=right, 2=bottom, 3=left
                if border_type == 0:
                    center_y = np.random.randint(0, max_cluster_size)
                    center_x = np.random.randint(0, w)
                elif border_type == 1:
                    center_y = np.random.randint(0, h)
                    center_x = np.random.randint(w - max_cluster_size, w)
                elif border_type == 2:
                    center_y = np.random.randint(h - max_cluster_size, h)
                    center_x = np.random.randint(0, w)
                else:
                    center_y = np.random.randint(0, h)
                    center_x = np.random.randint(0, max_cluster_size)
            else:
                # Random position from target pixels
                idx = np.random.randint(0, len(target_y))
                center_y, center_x = target_y[idx], target_x[idx]
            
            # Random cluster size for this instance
            cluster_size = np.random.randint(max(1, max_cluster_size//3), max_cluster_size+1)
            
            for dy in range(-cluster_size, cluster_size+1):
                    for dx in range(-cluster_size, cluster_size+1):
                        # Random fill with higher probability near center
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist <= cluster_size and np.random.random() < (1.0 - dist/cluster_size):
                            x = center_x + dx
                            y = center_y + dy
                            if 0 <= x < w and 0 <= y < h and target_mask[y, x]:
                                result[y, x] = self.value
        
        return result
    
    def get_name(self):
        d_str = f"{self.density}" if not isinstance(self.density, tuple) else f"{self.density[0]}-{self.density[1]}"
        c_str = f"{self.cluster_size}" if not isinstance(self.cluster_size, tuple) else f"{self.cluster_size[0]}-{self.cluster_size[1]}"
        target_str = " (on black)" if self.target_value == 0 else " (on white)" if self.target_value == 1 else ""
        return f"Clustered Noise (d={d_str}, c={c_str}){target_str}"

class LinePatternCommand(ImageCommand):
    """Adds patterns of lines like grids, stripes, or random directions"""
    def __init__(self, 
                 pattern_type='random',  # 'grid', 'horizontal', 'vertical', 'diagonal', 'random'
                 line_count=(5, 15),
                 line_thickness=(2, 4),
                 line_length=(20, 60),
                 gap=(5, 15),          # Space between lines for grid/stripe patterns
                 value=0.0,            # 0=black lines, 1=white lines
                 probability=1.0):
        super().__init__(probability)
        self.pattern_type = pattern_type
        self.line_count = line_count
        self.line_thickness = line_thickness
        self.line_length = line_length
        self.gap = gap
        self.value = value
        
    def _apply(self, image):
        result = image.copy()
        h, w = image.shape[:2]
        
        # Convert to appropriate format for cv2
        img_normalized = (result * 255).astype(np.uint8) if result.max() <= 1.0 else result.astype(np.uint8)
        
        # Sample parameters
        actual_line_count = int(self._sample_param(self.line_count))
        actual_thickness = int(self._sample_param(self.line_thickness))
        actual_length = int(self._sample_param(self.line_length))
        actual_gap = int(self._sample_param(self.gap))
        
        # Convert value to uint8 format
        line_value = int(self.value * 255) if result.max() <= 1.0 else int(self.value)
        
        if self.pattern_type == 'grid':
            # Draw horizontal lines
            for i in range(0, h, actual_gap):
                cv2.line(img_normalized, (0, i), (w, i), line_value, actual_thickness)
            
            # Draw vertical lines
            for i in range(0, w, actual_gap):
                cv2.line(img_normalized, (i, 0), (i, h), line_value, actual_thickness)
                
        elif self.pattern_type == 'horizontal':
            # Draw horizontal stripes
            for i in range(0, h, actual_gap):
                cv2.line(img_normalized, (0, i), (w, i), line_value, actual_thickness)
                
        elif self.pattern_type == 'vertical':
            # Draw vertical stripes
            for i in range(0, w, actual_gap):
                cv2.line(img_normalized, (i, 0), (i, h), line_value, actual_thickness)
                
        elif self.pattern_type == 'diagonal':
            # Draw diagonal lines
            offset = h + w
            for i in range(-offset, offset, actual_gap):
                cv2.line(img_normalized, (0, i), (i, 0), line_value, actual_thickness)
                cv2.line(img_normalized, (i, h), (w, i), line_value, actual_thickness)
                
        else:  # 'random'
            # Draw random lines with more density
            for _ in range(actual_line_count):
                # Generate random start point
                pt1_x = np.random.randint(0, w)
                pt1_y = np.random.randint(0, h)
                
                # Generate random angle
                angle = np.random.uniform(0, 2 * np.pi)
                
                # Calculate end point based on length and angle
                pt2_x = int(pt1_x + actual_length * np.cos(angle))
                pt2_y = int(pt1_y + actual_length * np.sin(angle))
                
                # Draw the line
                cv2.line(img_normalized, (pt1_x, pt1_y), (pt2_x, pt2_y), 
                         line_value, actual_thickness)
        
        # Return to original range
        return img_normalized.astype(float) / 255.0 if result.max() <= 1.0 else img_normalized.astype(float)
    
    def get_name(self):
        t_str = f"{self.line_thickness}" if not isinstance(self.line_thickness, tuple) else f"{self.line_thickness[0]}-{self.line_thickness[1]}"
        return f"{self.pattern_type.capitalize()} Lines (t={t_str})"

class ParallelNoiseCommand(ImageCommand):
    """Applies multiple noise effects in parallel and combines results"""
    def __init__(self, 
                 add_commands=None,    # Commands that add content (salt, clusters, etc.)
                 remove_commands=None, # Commands that remove content (pepper, erosion, etc.)
                 add_weight=0.5,       # Weight for add operations (0-1)
                 remove_weight=0.5,    # Weight for remove operations (0-1)
                 probability=1.0):
        super().__init__(probability)
        self.add_commands = add_commands if add_commands else []
        self.remove_commands = remove_commands if remove_commands else []
        self.add_weight = add_weight
        self.remove_weight = remove_weight
    
    def _apply(self, image):
        # Keep the original image
        result = image.copy()
        
        # Apply "add" commands in parallel and combine with MAX
        if self.add_commands:
            add_results = [cmd.execute(image) for cmd in self.add_commands]
            if add_results:
                add_combined = np.maximum.reduce(add_results)
                # Apply with weight
                result = result * (1 - self.add_weight) + add_combined * self.add_weight
        
        # Apply "remove" commands in parallel and combine with MIN
        if self.remove_commands:
            remove_results = [cmd.execute(result) for cmd in self.remove_commands]
            if remove_results:
                remove_combined = np.minimum.reduce(remove_results)
                # Apply with weight
                result = result * (1 - self.remove_weight) + remove_combined * self.remove_weight
        
        return result
    
    def get_name(self):
        add_names = [cmd.get_name() for cmd in self.add_commands]
        remove_names = [cmd.get_name() for cmd in self.remove_commands]
        
        add_str = f"Add:[{', '.join(add_names)}]" if add_names else ""
        remove_str = f"Remove:[{', '.join(remove_names)}]" if remove_names else ""
        
        if add_str and remove_str:
            return f"Parallel({add_str}, {remove_str})"
        return f"Parallel({add_str}{remove_str})"

class RandomPixelFlipCommand(ImageCommand):
    """Randomly flips a specified percentage of pixels in the image"""
    def __init__(self, 
                 flip_percentage=(0.1, 0.3),  # Percentage of pixels to flip
                 target_value=None,           # Value of pixels to target (0=black areas, 1=white areas, None=all)
                 probability=1.0):
        super().__init__(probability)
        self.flip_percentage = flip_percentage
        self.target_value = target_value
        
    def _apply(self, image):
        # Sample parameters
        actual_flip_percentage = self._sample_param(self.flip_percentage)
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # Create target mask based on target_value
        if self.target_value is not None:
            if self.target_value == 1.0:
                target_mask = (image > 0.5)  # Target white pixels
            else:
                target_mask = (image < 0.5)  # Target black pixels
        else:
            target_mask = np.ones((h, w), dtype=bool)  # Target all pixels
            
        if np.sum(target_mask) == 0:  # No pixels to target
            return result
            
        # Get coordinates of targetable pixels
        target_y, target_x = np.where(target_mask)
        if len(target_y) == 0:
            return result
        
        # Calculate number of pixels to flip
        num_pixels_to_flip = int(len(target_y) * actual_flip_percentage)
        
        # Randomly select pixels to flip
        if num_pixels_to_flip > 0:
            flip_indices = np.random.choice(len(target_y), num_pixels_to_flip, replace=False)
            
            # Flip selected pixels
            for idx in flip_indices:
                y, x = target_y[idx], target_x[idx]
                result[y, x] = 1.0 - result[y, x]  # Flip the pixel value
        
        return result
    
    def get_name(self):
        return "RandomPixelFlip"
    
class CommandChain:
    """Chain of commands to be applied in sequence"""
    def __init__(self, commands=None):
        self.commands = commands if commands else []
    
    def add_command(self, command):
        self.commands.append(command)
        return self
    
    def execute(self, image):
        result = image.copy()
        for command in self.commands:
            result = command.execute(result)
        return result
    
    def get_name(self):
        if not self.commands:
            return "Original"
        return " → ".join([cmd.get_name() for cmd in self.commands])
    
    
class SaltPepperNoiseCommand(ImageCommand):
    """Adds salt and pepper noise (random white and black pixels) to an image"""
    def __init__(self, density=(0.01, 0.05), cluster_size=(1, 3), salt_vs_pepper=0.5, probability=1.0):
        """
        Initialize the salt and pepper noise command
        
        Args:
            density: Amount of noise (proportion of image affected)
            cluster_size: Size of noise clusters (pixels)
            salt_vs_pepper: Ratio of white to black noise (0=all pepper, 1=all salt)
            probability: Chance this command will execute
        """
        super().__init__(probability)
        self.density = density
        self.cluster_size = cluster_size
        self.salt_vs_pepper = salt_vs_pepper
    
    def _apply(self, image):
        # Sample parameters from ranges if provided
        actual_density = self._sample_param(self.density)
        actual_cluster_size = int(self._sample_param(self.cluster_size))
        actual_salt_ratio = self._sample_param(self.salt_vs_pepper)
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # Determine number of noise clusters
        num_pixels = h * w
        num_clusters = int(num_pixels * actual_density / (actual_cluster_size**2))
        
        # Add noise clusters
        for _ in range(num_clusters):
            # Random center position for the cluster
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            
            # Random value (salt or pepper)
            is_salt = np.random.random() < actual_salt_ratio
            value = 1.0 if is_salt else 0.0
            
            # Create the cluster
            for dy in range(-actual_cluster_size//2, actual_cluster_size//2 + 1):
                for dx in range(-actual_cluster_size//2, actual_cluster_size//2 + 1):
                    x = center_x + dx
                    y = center_y + dy
                    
                    # Ensure we stay within image bounds
                    if 0 <= x < w and 0 <= y < h:
                        result[y, x] = value
        
        return result
    
    def get_name(self):
        d_str = f"{self.density}" if not isinstance(self.density, tuple) else f"{self.density[0]}-{self.density[1]}"
        c_str = f"{self.cluster_size}" if not isinstance(self.cluster_size, tuple) else f"{self.cluster_size[0]}-{self.cluster_size[1]}"
        return f"Salt & Pepper (d={d_str}, c={c_str})"

class RandomLinesCommand(ImageCommand):
    """Adds random lines to an image"""
    def __init__(self, num_lines=(3, 10), thickness=(1, 2), length=(5, 25), value=0.0, target_value=None, probability=1.0):
        """
        Initialize the random lines command
        
        Args:
            num_lines: Number of lines to add
            thickness: Line thickness in pixels
            length: Line length in pixels
            value: Pixel value for lines (0=black, 1=white)
            target_value: Value of pixels to target (0=black areas, 1=white areas, None=all)
            probability: Chance this command will execute
        """
        super().__init__(probability)
        self.num_lines = num_lines
        self.thickness = thickness
        self.length = length
        self.value = value
        self.target_value = target_value
    
    def _apply(self, image):
        # Sample parameters from ranges if provided
        actual_num_lines = int(self._sample_param(self.num_lines))
        actual_thickness = int(self._sample_param(self.thickness))
        actual_length = int(self._sample_param(self.length))
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # Create target mask based on target_value
        if self.target_value is not None:
            if self.target_value == 1.0:
                target_mask = (image > 0.5)  # Target white pixels
            else:
                target_mask = (image < 0.5)  # Target black pixels
        else:
            target_mask = np.ones((h, w), dtype=bool)  # Target all pixels
            
        if np.sum(target_mask) == 0:  # No pixels to target
            return result
            
        # Get coordinates of targetable pixels
        target_y, target_x = np.where(target_mask)
        if len(target_y) == 0:
            return result
        
        # Convert to appropriate format for cv2.line
        img_normalized = (result * 255).astype(np.uint8) if result.max() <= 1.0 else result.astype(np.uint8)
        
        # Create a temporary image for drawing
        temp_img = np.zeros((h, w), dtype=np.uint8)
        
        # Add random lines
        for _ in range(actual_num_lines):
            # Generate random start point from target pixels
            idx = np.random.randint(0, len(target_y))
            pt1_y, pt1_x = target_y[idx], target_x[idx]
            
            # Generate random angle
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Calculate end point based on length and angle
            pt2_x = int(pt1_x + actual_length * np.cos(angle))
            pt2_y = int(pt1_y + actual_length * np.sin(angle))
            
            # Draw the line
            cv2.line(temp_img, (pt1_x, pt1_y), (pt2_x, pt2_y), 255, actual_thickness)
        
        # Only apply lines where they intersect with target mask
        line_mask = (temp_img > 0)
        result[line_mask & target_mask] = self.value
        
        return result
    
    def get_name(self):
        n_str = f"{self.num_lines}" if not isinstance(self.num_lines, tuple) else f"{self.num_lines[0]}-{self.num_lines[1]}"
        t_str = f"{self.thickness}" if not isinstance(self.thickness, tuple) else f"{self.thickness[0]}-{self.thickness[1]}"
        l_str = f"{self.length}" if not isinstance(self.length, tuple) else f"{self.length[0]}-{self.length[1]}"
        target_str = " (on black)" if self.target_value == 0 else " (on white)" if self.target_value == 1 else ""
        return f"Random Lines (n={n_str}, t={t_str}, l={l_str}){target_str}"