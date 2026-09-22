
import numpy as np
from modules.bbox_gen.utils.mesh import change_pcd_range


class BoundsTokenizerDiag:
    def __init__(self, bins, BOS_id, EOS_id, PAD_id):
        self.bins = bins
        self.BOS_id = BOS_id
        self.EOS_id = EOS_id
        self.PAD_id = PAD_id
    
    def encode(self, data_dict, coord_rg=(-1,1)):
        """
        Encode bounding boxes to token sequence

        Args:
            data_dict: dictionary containing bounding boxes
            coord_rg: range of coordinate values
        Returns:
            token sequence
        """
        bounds = data_dict["bounds"] # (s, 2, 3)

        all_vertices = bounds.reshape(-1, 6)

        all_vertices = change_pcd_range(all_vertices, from_rg=coord_rg, to_rg=(0.5/self.bins, 1-0.5/self.bins))
        quantized_vertices = (all_vertices * self.bins).astype(np.int32)
        
        tokens = []
        tokens.append(self.BOS_id)
        tokens.extend(quantized_vertices.flatten().tolist())
        tokens.append(self.EOS_id)
        tokens = np.array(tokens)

        return tokens
    
    def decode(self, tokens, coord_rg=(-1,1)):
        """
        Decode token sequence back to bounding boxes

        Args:
            tokens: token sequence
        Returns:
            bounding box array [N, 2, 3]
        """
        # Remove special tokens
        valid_tokens = []
        for t in tokens:
            if t != self.BOS_id and t != self.EOS_id and t != self.PAD_id:
                valid_tokens.append(t)
        
        # Ensure correct number of tokens (2 vertices per box, 3 coordinates per vertex)
        if len(valid_tokens) % (2 * 3) != 0:
            raise ValueError(f"Invalid token count: {len(valid_tokens)}")
        
        # Reshape to vertex coordinates
        points = np.array(valid_tokens).reshape(-1, 2, 3)
        
        # Convert quantized coordinates back to continuous values
        points = points / self.bins
        points = change_pcd_range(points, from_rg=(0.5/self.bins, 1-0.5/self.bins), to_rg=coord_rg)

        return points