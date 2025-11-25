#ifndef NVCOMP_WRAPPER_H
#define NVCOMP_WRAPPER_H

#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

/**
 * Enumeration of nvCOMP compression algorithms
 */
enum class nvcompAlgorithm {
    LZ4,
    SNAPPY,
    GDEFLATE,
    DEFLATE,
    ZSTANDARD,
    CASCADED,
    BITCOMP,
    ANS
};

/**
 * Result structure for nvCOMP compression
 */
struct nvcompCompressionResult {
    size_t compressed_size_bytes;           // Size of compressed data
    size_t original_size_bytes;             // Size of original data
    double compression_time_ms;             // Compression time in milliseconds
    bool success;                           // Success flag
    std::string error_message;              // Error message if failed
};

/**
 * Result structure for nvCOMP decompression
 */
struct nvcompDecompressionResult {
    size_t decompressed_size_bytes;         // Size of decompressed data
    double decompression_time_ms;           // Decompression time in milliseconds
    bool success;                           // Success flag
    std::string error_message;              // Error message if failed
};

/**
 * Compress data on GPU using specified nvCOMP algorithm
 * @param d_input_data Device pointer to input data
 * @param input_size_bytes Size of input data in bytes
 * @param algorithm nvCOMP algorithm to use
 * @param d_output_data Device pointer to output buffer (must be pre-allocated with sufficient size)
 * @param max_output_size_bytes Maximum size of output buffer
 * @return nvcompCompressionResult containing compression metrics
 */
nvcompCompressionResult compress_nvcomp(
    const uint8_t* d_input_data,
    size_t input_size_bytes,
    nvcompAlgorithm algorithm,
    uint8_t* d_output_data,
    size_t max_output_size_bytes
);

/**
 * Decompress data on GPU using specified nvCOMP algorithm
 * @param d_compressed_data Device pointer to compressed data
 * @param compressed_size_bytes Size of compressed data in bytes
 * @param original_size_bytes Expected size of decompressed data
 * @param algorithm nvCOMP algorithm to use
 * @param d_output_data Device pointer to output buffer (must be pre-allocated with original_size_bytes)
 * @return nvcompDecompressionResult containing decompression metrics
 */
nvcompDecompressionResult decompress_nvcomp(
    const uint8_t* d_compressed_data,
    size_t compressed_size_bytes,
    size_t original_size_bytes,
    nvcompAlgorithm algorithm,
    uint8_t* d_output_data
);

/**
 * Get maximum compressed size for given input size and algorithm
 */
size_t nvcomp_get_max_compressed_size(size_t input_size_bytes, nvcompAlgorithm algorithm);

/**
 * Get algorithm name as string
 */
std::string nvcomp_algorithm_name(nvcompAlgorithm algorithm);

/**
 * Get all available algorithms
 */
std::vector<nvcompAlgorithm> get_all_nvcomp_algorithms();

#endif // NVCOMP_WRAPPER_H

