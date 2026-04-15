from microweaver.visualization.config import VisualizationConfig
from microweaver.visualization.report_visualize.chart_visualize import main as chart_main
from microweaver.visualization.report_visualize.table_visualize import main as table_main
from microweaver.visualization.graph_visualize.generate_graph import main as graph_main


def main():
    config = VisualizationConfig()
    table_main(config)
    chart_main(config)
    graph_main(config)


if __name__ == "__main__":
    main()