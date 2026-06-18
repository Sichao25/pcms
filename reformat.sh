while read file; do
  clang-format -i $file
done < <(find . -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx")
