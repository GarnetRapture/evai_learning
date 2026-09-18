set_project("lfm2_kernels")
set_xmakever("3.1.1")
add_rules("mode.release")

target("lfm2_kernels_cuda")
    set_kind("shared")
    add_toolchains("cuda")
    set_languages("c++20")
    set_runtimes("MD")
    add_files("*.cu")
    add_headerfiles("*.cuh")
    add_cugencodes("sm_86", "compute_86")
    add_cuflags(
        "--expt-relaxed-constexpr",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-Xcompiler=/Zc:__cplusplus",
        "-Xcompiler=/Zc:preprocessor",
        {force = true})
    add_defines("NOMINMAX")
    add_links("c10", "c10_cuda", "torch", "torch_cpu", "torch_cuda", "cudart_static")
    on_load(function (target)
        local query = os.iorunv("python", {
            "-c",
            "import pathlib,torch;r=pathlib.Path(torch.__file__).parent;"
                .. "print(r/'include');print(r/'include'/'torch'/'csrc'/'api'/'include');print(r/'lib')"})
        local lines = query:split("\n")
        target:add("cuflags", "-isystem=" .. lines[1]:trim(), {force = true})
        target:add("cuflags", "-isystem=" .. lines[2]:trim(), {force = true})
        target:add("linkdirs", lines[3]:trim())
    end)
    after_build(function (target)
        os.cp(target:targetfile(), path.join(os.scriptdir(), ".."))
    end)
