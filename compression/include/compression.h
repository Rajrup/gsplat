#ifndef COMPRESSION_H
#define COMPRESSION_H

#include <string>
#include <vector>
#include <cstdint>
#include "nvcomp_wrapper.h"
#include "utils.h"

/**
 * Compress point cloud using PCL octree compression (geometry only)
 * @param points Vector of 3D points (geometry only)
 * @return CompressionResult containing compressed data and compression time
 */
CompressionResult compress_pcl(const std::vector<Point3D>& points);

/**
 * Decompress point cloud using PCL octree decompression
 * @param compressed_data Compressed data buffer
 * @param output_path Path to save decompressed PLY file
 * @return DecompressionResult indicating success/failure
 */
DecompressionResult decompress_pcl(const std::vector<uint8_t>& compressed_data, const std::string& output_path);

/**
 * Compress point cloud using Open3D octree compression (geometry only)
 * @param points Vector of 3D points (geometry only)
 * @return CompressionResult containing compressed data and compression time
 */
CompressionResult compress_open3d(const std::vector<Point3D>& points);

/**
 * Decompress point cloud using Open3D octree decompression
 * @param compressed_data Compressed data buffer
 * @param output_path Path to save decompressed PLY file
 * @return DecompressionResult indicating success/failure
 */
DecompressionResult decompress_open3d(const std::vector<uint8_t>& compressed_data, const std::string& output_path);

/**
 * Compress point cloud using Draco compression (geometry only)
 * @param points Vector of 3D points (geometry only)
 * @return CompressionResult containing compressed data and compression time
 */
CompressionResult compress_draco(const std::vector<Point3D>& points);

/**
 * Decompress point cloud using Draco decompression
 * @param compressed_data Compressed data buffer
 * @param output_path Path to save decompressed PLY file
 * @return DecompressionResult indicating success/failure
 */
DecompressionResult decompress_draco(const std::vector<uint8_t>& compressed_data, const std::string& output_path);

/**
 * Compress point cloud using GPU-optimized octree compression
 * @param points Vector of 3D points (geometry only)
 * @param octree_depth Depth of the octree (e.g., 10 for 1024x1024x1024 grid)
 * @param nvcomp_algorithm Optional nvCOMP algorithm for lossless compression. If nullptr, returns uncompressed serialized bytestream.
 * @return CompressionResult containing compressed data and compression time
 *         If nvcomp_algorithm is provided, compressed_data format: [serialized_size (8 bytes)][nvcomp_compressed_data...]
 *         If nvcomp_algorithm is nullptr, compressed_data format: [num_levels][level_sizes...][bfs_stream...]
 */
CompressionResult compress_gpu_octree(
    const std::vector<Point3D>& points, 
    uint32_t octree_depth,
    nvcompAlgorithm* nvcomp_algorithm = nullptr
);

/**
 * Decompress point cloud using GPU-optimized octree decompression
 * @param compressed_data Compressed data buffer
 * @param output_path Path to save decompressed PLY file
 * @param octree_depth Expected depth of the octree (must match compression depth)
 * @param nvcomp_algorithm Optional nvCOMP algorithm used for compression. If nullptr, assumes uncompressed serialized bytestream.
 * @return DecompressionResult indicating success/failure
 */
DecompressionResult decompress_gpu_octree(
    const std::vector<uint8_t>& compressed_data, 
    const std::string& output_path, 
    uint32_t octree_depth,
    nvcompAlgorithm* nvcomp_algorithm = nullptr
);

#endif // COMPRESSION_H

