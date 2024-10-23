import tensorrt as trt


def convert_onnx_to_tensorrt(onnx_model_path, engine_file_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # 使用显式批次模式创建网络
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)

    # 加载 ONNX 模型
    with open(onnx_model_path, 'rb') as f:
        onnx_model = f.read()

    # 解析 ONNX 模型
    if not parser.parse(onnx_model):
        print('Failed to parse the ONNX model.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return

    # 使用 BuilderConfig 设置参数
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # 创建网络并构建 TensorRT 引擎
    serialized_network = builder.build_serialized_network(network, config)
    if serialized_network is None:
        print('Failed to build the TensorRT engine.')
        return

    # 保存引擎到文件
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_network)

    print(f'Engine saved to {engine_file_path}')


# 示例用法
convert_onnx_to_tensorrt('E:/WorkSpace/depth_image/best_train4.onnx', 'E:/WorkSpace/depth_image/best_train4.engine')





