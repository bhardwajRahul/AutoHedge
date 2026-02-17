import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel
from swarms import Conversation

from autohedge.workers import (
    ExecutionAgent,
    QuantAnalyst,
    RiskManager,
    TradingDirector,
)


class AutoHedgeOutput(BaseModel):
    id: str = uuid.uuid4().hex
    thesis: Optional[str] = None
    risk_assessment: Optional[str] = None
    order: Optional[str] = None
    decision: str = None
    timestamp: str = datetime.now().isoformat()
    current_stock: str


class AutoHedgeOutputMain(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    id: str = uuid.uuid4().hex
    stocks: Optional[list] = None
    task: Optional[str] = None
    timestamp: str = datetime.now().isoformat()
    logs: List[AutoHedgeOutput] = None


class AutoHedge:
    """
    Main trading system that coordinates all agents and manages the trading cycle.

    Attributes:
        stocks (List[str]): List of stock tickers to trade
        director (TradingDirector): Trading director agent
        quant (QuantAnalyst): Quantitative analysis agent
        risk (RiskManager): Risk management agent
        execution (ExecutionAgent): Trade execution agent
        output_dir (Path): Directory for storing outputs
    """

    def __init__(
        self,
        stocks: List[str],
        name: str = "autohedge",
        description: str = "fully autonomous hedgefund",
        output_dir: str = "outputs",
        output_file_path: str = None,
        strategy: str = None,
        output_type: str = "list",
    ):
        """
        Initialize the AutoHedge class.

        Args:
            stocks (List[str]): List of stock tickers to trade
            name (str, optional): Name of the trading system. Defaults to "autohedge".
            description (str, optional): Description of the trading system. Defaults to "fully autonomous hedgefund".
            output_dir (str, optional): Directory for storing outputs. Defaults to "outputs".
            output_file_path (str, optional): Path to the output file. Defaults to None.
        """
        self.name = name
        self.description = description
        self.stocks = stocks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.strategy = strategy
        self.output_type = output_type
        self.output_file_path = output_file_path
        logger.info("Initializing Automated Trading System")
        self.director = TradingDirector(stocks, output_dir)
        self.quant = QuantAnalyst()
        self.risk = RiskManager()
        self.execution = ExecutionAgent()
        self.logs = AutoHedgeOutputMain(
            name=self.name,
            description=self.description,
            stocks=stocks,
            task="",
            logs=[],
        )
        self.conversation = Conversation(time_enabled=True)

    def run(self, task: str, *args, **kwargs):
        """
        Execute one complete trading cycle for all stocks.

        Args:
            task (str): The task to be executed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List: List of logs for each stock.
        """
        logger.info("Starting trading cycle")
        self.conversation.add(role="user", content=f"Task: {task}")

        try:
            for stock in self.stocks:
                logger.info(f"Processing {stock}")

                # Generate thesis
                thesis, market_data = self.director.generate_thesis(
                    task=task, stock=stock
                )

                self.conversation.add(
                    role=self.director.agent_name,
                    content=f"Stock: {stock}\nMarket Data: {market_data}\nThesis: {thesis}",
                )

                # Perform analysis
                analysis = self.quant.analyze(
                    stock + market_data, thesis, task=task
                )

                # setiment_analysis = sentiment_agent.run(
                #     fetch_stock_news(stock)
                # )

                # logger.info(f"Sentiment Analysis: {setiment_analysis}")

                # self.conversation.add(sentiment_agent.agent_name, setiment_analysis)

                self.conversation.add(
                    role=self.quant.agent_name, content=analysis
                )

                # Assess risk
                risk_assessment = self.risk.assess_risk(
                    stock + market_data, thesis, analysis, task=task
                )

                self.conversation.add(
                    role=self.risk.agent_name, content=risk_assessment
                )

                # # Generate order if approved
                order = self.execution.generate_order(
                    stock, thesis, risk_assessment, task=task
                )

                self.conversation.add(
                    role=self.execution.agent_name, content=order
                )

                order = str(order)

                # Final decision
                decision = self.director.make_decision(
                    order + market_data + risk_assessment,
                    thesis,
                    user_task=task,
                )

                self.conversation.add(
                    role=self.director.agent_name, content=decision
                )

            #     log = AutoHedgeOutput(
            #         thesis=thesis,
            #         risk_assessment=risk_assessment,
            #         current_stock=stock,
            #         order=order,
            #         decision=decision,
            #     )

            #     # logs.append(log.model_dump_json(indent=4))
            #     self.logs.task = task
            #     self.logs.logs.append(log)

            # create_file_in_folder(
            #     self.output_dir,
            #     f"analysis-{uuid.uuid4().hex}.json",
            #     self.logs.model_dump_json(indent=4),
            # )

            # return self.logs.model_dump_json(indent=4)
            if self.output_type == "list":
                return self.conversation.return_messages_as_list()
            elif self.output_type == "dict":
                return (
                    self.conversation.return_messages_as_dictionary()
                )
            elif self.output_type == "str":
                return self.conversation.return_history_as_string()

        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            raise
