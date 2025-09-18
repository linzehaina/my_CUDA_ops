## Nsight System使用

### （1）通过命令行采集分析数据

在终端中使用 `nsys` 命令可以对你的程序进行性能分析。

```bash
nsys profile --stats=true -o report_name ./your_program
```

- `profile` 是 Nsight Systems 的主要命令，表示进行性能分析。
- `--stats=true` 表示在分析后打印统计信息。
- `-o report_name` 表示生成的报告文件名。
- `./your_program` 是要分析的可执行程序。

这个命令会生成nsys-rep，qdrep，sqlite三种格式的报告文件，包含详细的性能数据。

### （2）使用nsys-ui打开GUI界面，查看刚刚生成的report_name.nsys-rep文件



## Nsight Compute使用

```bash
ncu --set full --target-processes all -o my_report ./your_program
```

- `--set full`：执行全面的 CUDA 内核分析，包含内存使用、执行效率等详细的性能指标。
- `--target-processes all`：捕获程序中所有的 CUDA 内核执行信息。
- `./your_program`：你想要分析的 CUDA 程序。