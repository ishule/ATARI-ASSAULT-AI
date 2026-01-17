/*
 * reduce_noop.cpp
 *
 * Reduce a percentage of "no-op" rows from a CSV-like dataset where fields are
 * separated by ';' and the last 3 columns are expected to be SPACE;LEFT;RIGHT.
 *
 * Usage:
 *   ./reduce_noop input.csv 50
 *   ./reduce_noop input.csv 50 -o out.csv
 *   ./reduce_noop input.csv 50 -o out.csv -s 42
 *   ./reduce_noop input.csv 50 --header
 *
 * Options:
 *   -o, --output <file>   Output filename. If omitted, defaults to
 *                         <input_basename>_reduced_<pct>.csv
 *   -s, --seed <int>      Seed for RNG to make selection reproducible.
 *   --header              Preserve the first non-empty line as a header;
 *                         it will not be considered for reduction.
 *
 * Notes:
 * - A "no-op" row is a row where the last three fields all parse to 0.
 * - The program preserves the original order of rows (except for removed rows).
 * - Empty lines in the input are ignored.
 *
 * Build:
 *   g++ -std=c++17 -O2 -o reduce_noop scripts/reduce_noop.cpp
 *
 */

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <unordered_set>

using namespace std;

static const char SEP = ';';

static inline string trim_copy(const string &s) {
    size_t a = 0;
    while (a < s.size() && isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static inline vector<string> split_fields(const string &line, char sep = SEP) {
    vector<string> out;
    string cur;
    for (char ch : line) {
        if (ch == sep) {
            out.push_back(trim_copy(cur));
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    // push last field (even if empty)
    out.push_back(trim_copy(cur));
    // remove trailing empty fields created by final separators
    while (!out.empty() && out.back().empty()) out.pop_back();
    return out;
}

static inline bool safe_int_zero(const string &s) {
    if (s.empty()) return true; // treat empty as zero (robust)
    // allow "0", "00", maybe spaces already trimmed
    // try to parse as integer; treat non-int as not-zero (conservative)
    try {
        size_t idx = 0;
        long v = stol(s, &idx, 10);
        (void)v;
        if (idx != s.size()) return false;
        return (v == 0);
    } catch (...) {
        return false;
    }
}

bool is_noshot_row(const string &line) {
    auto fields = split_fields(line, SEP);
    if (fields.size() < 3) return false;
    // look at last 3 fields
    size_t n = fields.size();
    return safe_int_zero(fields[n - 3]) && (!safe_int_zero(fields[n - 2]) || !safe_int_zero(fields[n - 1]));
}

void usage(const char *prog) {
    cerr << "Usage: " << prog << " <input.csv> <percent> [-o out.csv] [-s seed] [--header]\n";
    cerr << "  percent: 0-100 percentage of no-op rows to remove (e.g. 50)\n";
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    string input = argv[1];
    double percent = 0.0;
    try {
        percent = stod(argv[2]);
    } catch (...) {
        cerr << "Invalid percent: " << argv[2] << "\n";
        return 1;
    }
    if (!(percent >= 0.0 && percent <= 100.0)) {
        cerr << "Percent must be between 0 and 100\n";
        return 1;
    }

    string output;
    bool have_seed = false;
    unsigned int seed = 0;
    bool has_header = false;

    // parse optional args
    for (int i = 3; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output = argv[++i];
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            have_seed = true;
            seed = static_cast<unsigned int>(stoul(argv[++i]));
        } else if (arg == "--header") {
            has_header = true;
        } else {
            cerr << "Unknown option: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (output.empty()) {
        // default output name
        filesystem::path p(input);
        string base = p.stem().string();
        output = base + "_reduced_" + to_string(static_cast<int>(percent)) + ".csv";
    }

    // Read input lines, skip empty lines but preserve order
    ifstream ifs(input);
    if (!ifs) {
        cerr << "Cannot open input file: " << input << "\n";
        return 1;
    }

    vector<string> lines;
    string line;
    while (std::getline(ifs, line)) {
        // preserve non-empty lines only (trim whitespace to decide)
        bool all_ws = true;
        for (char c : line) {
            if (!isspace(static_cast<unsigned char>(c))) { all_ws = false; break; }
        }
        if (!all_ws) lines.push_back(line);
    }
    ifs.close();

    if (lines.empty()) {
        cerr << "No non-empty lines found in input.\n";
        return 1;
    }

    // If header flag, treat first non-empty line as header and don't consider it for reduction
    int header_index = -1;
    if (has_header) header_index = 0;

    // Find indices of noshot rows (excluding header if present)
    vector<int> noshot_indices;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (static_cast<int>(i) == header_index) continue;
        if (is_noshot_row(lines[i])) noshot_indices.push_back(static_cast<int>(i));
    }

    const int total_rows = static_cast<int>(lines.size());
    const int total_noshot = static_cast<int>(noshot_indices.size());
    const int remove_count = static_cast<int>(round(total_noshot * (percent / 100.0)));

    // select random subset to remove
    vector<int> to_remove;
    if (remove_count > 0 && total_noshot > 0) {
        std::mt19937 rng;
        if (have_seed) rng.seed(seed);
        else {
            std::random_device rd;
            rng.seed(rd());
        }
        // shuffle indices and take first remove_count
        shuffle(noshot_indices.begin(), noshot_indices.end(), rng);
        to_remove.assign(noshot_indices.begin(), noshot_indices.begin() + remove_count);
    }
    // make set for fast lookup
    unordered_set<int> remove_set(to_remove.begin(), to_remove.end());

    // Write output preserving order (and header line)
    ofstream ofs(output);
    if (!ofs) {
        cerr << "Cannot open output file for writing: " << output << "\n";
        return 1;
    }

    int written = 0;
    int removed = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (remove_set.count(static_cast<int>(i))) {
            ++removed;
            continue;
        }
        ofs << lines[i] << '\n';
        ++written;
    }
    ofs.close();

    cout << "Reduction summary:\n";
    cout << "  input file:            " << input << '\n';
    cout << "  output file:           " << output << '\n';
    cout << "  total rows (non-empty):" << total_rows << '\n';
    cout << "  detected no-shot rows:   " << total_noshot << '\n';
    cout << "  percent requested:     " << percent << "%\n";
    cout << "  no-shot rows removed:    " << removed << '\n';
    cout << "  rows written:          " << written << '\n';

    return 0;
}