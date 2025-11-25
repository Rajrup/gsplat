#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <set>
#include <cuda_runtime.h>
#include "utils.h"
#include "compression.h"

using namespace std;
namespace fs = std::filesystem;

// Helper function to validate lossless compression
bool validate_lossless_compression(const vector<Point3D>& original, const string& decompressed_path) {
    // Read decompressed points
    vector<Point3D> decompressed = read_ply_geometry(decompressed_path);

    if (decompressed.empty()) {
        cerr << "  [VALIDATION FAILED] Could not read decompressed file" << endl;
        return false;
    }

    // Create sets for comparison (octree compression deduplicates identical voxel positions)
    set<Point3D> original_set(original.begin(), original.end());
    set<Point3D> decompressed_set(decompressed.begin(), decompressed.end());

    // Check if sets match
    if (original_set == decompressed_set) {
        cout << "  [VALIDATION PASSED] Compression is LOSSLESS" << endl;
        cout << "  Original unique points: " << original_set.size() << endl;
        cout << "  Decompressed points: " << decompressed_set.size() << endl;
        if (original.size() != original_set.size()) {
            cout << "  Note: Original had " << (original.size() - original_set.size())
                      << " duplicate voxel positions (expected for voxelized data)" << endl;
        }
        return true;
    } else {
        cerr << "  [VALIDATION FAILED] Point sets do not match!" << endl;
        cerr << "  Original unique points: " << original_set.size() << endl;
        cerr << "  Decompressed points: " << decompressed_set.size() << endl;

        // Find differences
        set<Point3D> missing_in_decompressed;
        set_difference(original_set.begin(), original_set.end(),
                           decompressed_set.begin(), decompressed_set.end(),
                           inserter(missing_in_decompressed, missing_in_decompressed.begin()));

        set<Point3D> extra_in_decompressed;
        set_difference(decompressed_set.begin(), decompressed_set.end(),
                           original_set.begin(), original_set.end(),
                           inserter(extra_in_decompressed, extra_in_decompressed.begin()));

        if (!missing_in_decompressed.empty()) {
            cerr << "  Missing in decompressed: " << missing_in_decompressed.size() << " points" << endl;
        }
        if (!extra_in_decompressed.empty()) {
            cerr << "  Extra in decompressed: " << extra_in_decompressed.size() << " points" << endl;
        }

        return false;
    }
}

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " -i <input> -o <output_folder> [options]\n\n";
    cout << "Required arguments:\n";
    cout << "  -i, --input <path>       Input PLY file or folder containing PLY files\n";
    cout << "  -o, --output <folder>    Output base folder for compressed results\n\n";
    cout << "Optional arguments:\n";
    cout << "  -n, --num <count>        Max number of files to process (default: 10, ignored for single file)\n";
    cout << "  -d, --device <device>    CUDA device to use (e.g., cuda:0, cuda:1) (default: cuda:0)\n";
    cout << "  -t, --depth <depth>      Octree depth for GPU octree compression (default: 10)\n";
    cout << "  -h, --help               Show this help message\n\n";
    cout << "Examples:\n";
    cout << "  " << program_name << " -i ./input.ply -o ./results\n";
    cout << "  " << program_name << " -i ./input_ply -o ./results -n 5\n";
    cout << "  " << program_name << " -i ./input_ply -o ./results -d cuda:1 -t 10\n\n";
    cout << "Note: PLY files must have point coordinates in [0, 1023] range.\n";
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    string input_path;
    string output_base;
    int max_files = 10;
    string device_str = "cuda:0";
    int octree_depth = 10;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                input_path = argv[++i];
            } else {
                cerr << "Error: -i requires an argument\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_base = argv[++i];
            } else {
                cerr << "Error: -o requires an argument\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-n" || arg == "--num") {
            if (i + 1 < argc) {
                max_files = stoi(argv[++i]);
            } else {
                cerr << "Error: -n requires an argument\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-d" || arg == "--device") {
            if (i + 1 < argc) {
                device_str = argv[++i];
            } else {
                cerr << "Error: -d requires an argument\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-t" || arg == "--depth") {
            if (i + 1 < argc) {
                octree_depth = stoi(argv[++i]);
                if (octree_depth < 1 || octree_depth > 10) {
                    cerr << "Error: Octree depth must be between 1 and 10\n";
                    return 1;
                }
            } else {
                cerr << "Error: -t requires an argument\n";
                print_usage(argv[0]);
                return 1;
            }
        } else {
            cerr << "Error: Unknown argument " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (input_path.empty() || output_base.empty()) {
        cerr << "Error: Missing required arguments\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Parse and set CUDA device
    int cuda_device = 0;
    if (device_str.substr(0, 5) == "cuda:") {
        try {
            cuda_device = stoi(device_str.substr(5));
        } catch (const exception& e) {
            cerr << "Error: Invalid device string format. Use 'cuda:N' where N is the device number.\n";
            return 1;
        }

        // Check if device is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            cerr << "Error: Failed to get CUDA device count: " << cudaGetErrorString(err) << "\n";
            return 1;
        }

        if (cuda_device >= device_count || cuda_device < 0) {
            cerr << "Error: CUDA device " << cuda_device << " is not available. ";
            cerr << "Available devices: 0-" << (device_count - 1) << "\n";
            return 1;
        }

        // Set CUDA device
        err = cudaSetDevice(cuda_device);
        if (err != cudaSuccess) {
            cerr << "Error: Failed to set CUDA device " << cuda_device << ": " << cudaGetErrorString(err) << "\n";
            return 1;
        }
    } else if (device_str == "cpu") {
        cerr << "Warning: CPU mode is not implemented. GPU octree compression requires CUDA.\n";
        cerr << "         GPU octree compression will be skipped.\n";
    } else {
        cerr << "Error: Invalid device string. Use 'cuda:N' or 'cpu'.\n";
        return 1;
    }

    // Validate input path exists
    if (!fs::exists(input_path)) {
        cerr << "Error: Input path does not exist: " << input_path << "\n";
        return 1;
    }

    // Determine if input is a file or directory
    bool is_single_file = fs::is_regular_file(input_path);
    bool is_directory = fs::is_directory(input_path);

    if (!is_single_file && !is_directory) {
        cerr << "Error: Input path must be a regular file or directory: " << input_path << "\n";
        return 1;
    }

    // Create output directories
    fs::create_directories(output_base + "/pcl");
    fs::create_directories(output_base + "/open3d");
    fs::create_directories(output_base + "/draco");
    fs::create_directories(output_base + "/gpu_octree");

    cout << "Input: " << input_path << (is_single_file ? " (file)" : " (folder)") << "\n";
    cout << "Output folder: " << output_base << "\n";
    if (!is_single_file) {
        cout << "Max files to process: " << max_files << "\n";
    }
    cout << "CUDA device: " << device_str << "\n";
    cout << "Octree depth: " << octree_depth << " (grid size: " << (1 << octree_depth) << "^3)\n\n";

    // Get list of PLY files
    vector<string> ply_files;

    if (is_single_file) {
        // Single file mode
        if (fs::path(input_path).extension() != ".ply") {
            cerr << "Error: Input file must have .ply extension: " << input_path << "\n";
            return 1;
        }
        ply_files.push_back(input_path);
    } else {
        // Directory mode
        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".ply") {
                ply_files.push_back(entry.path().string());
            }
        }

        if (ply_files.empty()) {
            cerr << "Error: No PLY files found in " << input_path << "\n";
            return 1;
        }

        // Sort files to get first N
        sort(ply_files.begin(), ply_files.end());
    }

    // Process first N files
    int num_files = min(max_files, static_cast<int>(ply_files.size()));
    
    cout << "Processing " << num_files << " PLY files..." << endl;
    cout << "=========================================" << endl;
    
    for (int i = 0; i < num_files; ++i) {
        string input_file = ply_files[i];
        fs::path input_path(input_file);
        string filename = input_path.stem().string(); // e.g., "redandblack_vox10_1450"
        
        // Extract point cloud number (last part after underscore)
        size_t last_underscore = filename.find_last_of('_');
        string ptcl_number = (last_underscore != string::npos) ? 
                                  filename.substr(last_underscore + 1) : filename;
        
        cout << "\nProcessing file " << (i + 1) << "/" << num_files << ": " << input_path.filename().string() << endl;
        cout << "Point cloud number: " << ptcl_number << endl;
        
        // Read point cloud geometry once
        cout << "Reading point cloud geometry..." << endl;
        vector<Point3D> points = read_ply_geometry(input_file);
        if (points.empty()) {
            cerr << "Error: Failed to read point cloud from " << input_file << endl;
            continue;
        }
        cout << "Loaded " << points.size() << " points" << endl;
        
        // Validate that all coordinates are within [0, 1023] range for voxelized coordinates
        for (const auto& pt : points) {
            assert(pt.x <= 1023u && pt.y <= 1023u && pt.z <= 1023u &&
                   "Point coordinates must be in [0, 1023] range for voxelized coordinates");
        }
        
        // PCL Compression
        cout << "\n--- PCL Compression ---" << endl;
        CompressionResult pcl_result = compress_pcl(points);
        if (pcl_result.compression_time_ms > 0) {
            cout << "Original size: " << pcl_result.original_size_bytes << " bytes (" 
                      << (pcl_result.original_size_bytes / 1024.0) << " KB)" << endl;
            cout << "Compressed size: " << pcl_result.compressed_size_bytes << " bytes (" 
                      << (pcl_result.compressed_size_bytes / 1024.0) << " KB)" << endl;
            cout << "Compression ratio: " << (double)pcl_result.original_size_bytes / pcl_result.compressed_size_bytes 
                      << ":1" << endl;
            cout << "Compression time: " << pcl_result.compression_time_ms << " ms" << endl;
            string pcl_output = output_base + "/pcl/" + ptcl_number + ".ply";
            DecompressionResult pcl_decomp = decompress_pcl(pcl_result.compressed_data, pcl_output);
            if (pcl_decomp.success) {
                cout << "Decompression time: " << pcl_decomp.decompression_time_ms << " ms" << endl;
                cout << "Decompressed and saved to: " << pcl_output << endl;
            } else {
                cerr << "Failed to decompress PCL" << endl;
            }
        } else {
            cerr << "Failed to compress with PCL" << endl;
        }
        
        // Open3D Compression
        cout << "\n--- Open3D Compression ---" << endl;
        CompressionResult open3d_result = compress_open3d(points);
        if (open3d_result.compression_time_ms > 0) {
            cout << "Original size: " << open3d_result.original_size_bytes << " bytes (" 
                      << (open3d_result.original_size_bytes / 1024.0) << " KB)" << endl;
            cout << "Compressed size: " << open3d_result.compressed_size_bytes << " bytes (" 
                      << (open3d_result.compressed_size_bytes / 1024.0) << " KB)" << endl;
            cout << "Compression ratio: " << (double)open3d_result.original_size_bytes / open3d_result.compressed_size_bytes 
                      << ":1" << endl;
            cout << "Compression time: " << open3d_result.compression_time_ms << " ms" << endl;
            string open3d_output = output_base + "/open3d/" + ptcl_number + ".ply";
            DecompressionResult open3d_decomp = decompress_open3d(open3d_result.compressed_data, open3d_output);
            if (open3d_decomp.success) {
                cout << "Decompression time: " << open3d_decomp.decompression_time_ms << " ms" << endl;
                cout << "Decompressed and saved to: " << open3d_output << endl;
            } else {
                cerr << "Failed to decompress Open3D" << endl;
            }
        } else {
            cerr << "Failed to compress with Open3D" << endl;
        }
        
        // Draco Compression
        cout << "\n--- Draco Compression ---" << endl;
        CompressionResult draco_result = compress_draco(points);
        if (draco_result.compression_time_ms > 0) {
            cout << "Original size: " << draco_result.original_size_bytes << " bytes (" 
                      << (draco_result.original_size_bytes / 1024.0) << " KB)" << endl;
            cout << "Compressed size: " << draco_result.compressed_size_bytes << " bytes (" 
                      << (draco_result.compressed_size_bytes / 1024.0) << " KB)" << endl;
            cout << "Compression ratio: " << (double)draco_result.original_size_bytes / draco_result.compressed_size_bytes 
                      << ":1" << endl;
            cout << "Compression time: " << draco_result.compression_time_ms << " ms" << endl;
            string draco_output = output_base + "/draco/" + ptcl_number + ".ply";
            DecompressionResult draco_decomp = decompress_draco(draco_result.compressed_data, draco_output);
            if (draco_decomp.success) {
                cout << "Decompression time: " << draco_decomp.decompression_time_ms << " ms" << endl;
                cout << "Decompressed and saved to: " << draco_output << endl;
            } else {
                cerr << "Failed to decompress Draco" << endl;
            }
        } else {
            cerr << "Failed to compress with Draco" << endl;
        }
        
        // GPU Octree Compression
        cout << "\n--- GPU Octree Compression ---" << endl;
        if (device_str != "cpu") {
            CompressionResult gpu_octree_result = compress_gpu_octree(points, octree_depth);
            if (gpu_octree_result.compression_time_ms > 0) {
                cout << "Original size: " << gpu_octree_result.original_size_bytes << " bytes ("
                          << (gpu_octree_result.original_size_bytes / 1024.0) << " KB)" << endl;
                cout << "Compressed size: " << gpu_octree_result.compressed_size_bytes << " bytes ("
                          << (gpu_octree_result.compressed_size_bytes / 1024.0) << " KB)" << endl;
                cout << "Compression ratio: " << (double)gpu_octree_result.original_size_bytes / gpu_octree_result.compressed_size_bytes
                          << ":1" << endl;
                cout << "Compression time: " << gpu_octree_result.compression_time_ms << " ms" << endl;
                string gpu_octree_output = output_base + "/gpu_octree/" + ptcl_number + ".ply";
                DecompressionResult gpu_octree_decomp = decompress_gpu_octree(gpu_octree_result.compressed_data, gpu_octree_output, octree_depth);
                if (gpu_octree_decomp.success) {
                    cout << "Decompression time: " << gpu_octree_decomp.decompression_time_ms << " ms" << endl;
                    cout << "Decompressed and saved to: " << gpu_octree_output << endl;

                    // Validate lossless compression
                    cout << "\n  Validating lossless compression..." << endl;
                    validate_lossless_compression(points, gpu_octree_output);
                } else {
                    cerr << "Failed to decompress GPU Octree" << endl;
                }
            } else {
                cerr << "Failed to compress with GPU Octree" << endl;
            }
        } else {
            cout << "Skipped (CPU mode selected)" << endl;
        }
        
        cout << "\n=========================================" << endl;
    }
    
    cout << "\nAll files processed!" << endl;
    return 0;
}

