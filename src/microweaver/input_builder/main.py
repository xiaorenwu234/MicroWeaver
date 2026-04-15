from microweaver.input_builder.config import InputConfig
from microweaver.input_builder.static_analyze.static_dependencies_mapper import main as static_main
from microweaver.input_builder.dynamic_analyze.dynamic_dependencies_mapper import main as dynamic_main
from microweaver.input_builder.merge import main as merge_main
from microweaver.input_builder.generate_description import main as generate_descriptions

import warnings

warnings.filterwarnings("ignore")


def main(config: InputConfig = InputConfig()):
    static_main(config)
    if config.merge_json:
        dynamic_main(config)
    if config.generate_description:
        generate_descriptions(config)


if __name__ == "__main__":
    main()
