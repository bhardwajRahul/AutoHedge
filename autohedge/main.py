import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from swarms import Agent, Conversation
from tickr_agent.main import TickrAgent

from autohedge.prompts import (
    DIRECTOR_PROMPT,
    EXECUTION_ORDER_PROMPT,
    EXECUTION_PROMPT,
    QUANT_ANALYSIS_PROMPT,
    QUANT_PROMPT,
    RISK_ASSESSMENT_PROMPT,
    RISK_PROMPT,
    SENTIMENT_PROMPT,
    DIRECTOR_THESIS_PROMPT,
    DIRECTOR_DECISION_PROMPT,
)

sentiment_agent = Agent(
    agent_name="Sentiment-Agent",
    system_prompt=SENTIMENT_PROMPT,
    model_name="gpt-4o-mini",
    output_type="str",
    max_loops=1,
    verbose=True,
    context_length=16000,
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


class RiskManager:
    def __init__(self):
        self.risk_agent = Agent(
            agent_name="Risk-Manager",
            system_prompt=RISK_PROMPT,
            model_name="groq/deepseek-r1-distill-llama-70b",
            output_type="str",
            max_loops=1,
            verbose=True,
            context_length=16000,
        )

    def assess_risk(
        self, stock: str, thesis: str, quant_analysis: str
    ) -> str:
        prompt = RISK_ASSESSMENT_PROMPT.format(
            stock=stock, thesis=thesis, quant_analysis=quant_analysis
        )
        assessment = self.risk_agent.run(prompt)

        return assessment


class ExecutionAgent:
    def __init__(self):
        self.execution_agent = Agent(
            agent_name="Execution-Agent",
            system_prompt=EXECUTION_PROMPT,
            model_name="groq/deepseek-r1-distill-llama-70b",
            output_type="str",
            max_loops=1,
            verbose=True,
            context_length=16000,
        )

    def generate_order(
        self, stock: str, thesis: Dict, risk_assessment: Dict
    ) -> str:
        prompt = EXECUTION_ORDER_PROMPT.format(
            stock=stock,
            thesis=thesis,
            risk_assessment=risk_assessment,
        )
        order = self.execution_agent.run(prompt)
        return order


class TradingDirector:
    """
    Trading Director Agent responsible for generating trading theses and coordinating strategy.

    Attributes:
        director_agent (Agent): Swarms agent for thesis generation
        tickr (TickrAgent): Agent for market data collection
        output_dir (Path): Directory for storing outputs

    Methods:
        generate_thesis: Generates trading thesis for a given stock
        save_output: Saves thesis to JSON file
    """

    def __init__(
        self,
        stocks: List[str],
        output_dir: str = "outputs",
        cryptos: List[str] = None,
    ):

        logger.info("Initializing Trading Director")
        self.director_agent = Agent(
            agent_name="Trading-Director",
            system_prompt=DIRECTOR_PROMPT,
            model_name="groq/deepseek-r1-distill-llama-70b",
            output_type="str",
            max_loops=1,
            verbose=True,
            context_length=16000,
        )

        # self.crypto_agent = CryptoAgentWrapper()

    def generate_thesis(
        self,
        task: str = "Generate a thesis for the stock",
        stock: str = None,
        crypto: str = None,
    ) -> str:
        """
        Generate trading thesis for a given stock.

        Args:
            stock (str): Stock ticker symbol

        Returns:
            TradingThesis: Generated thesis
        """
        logger.info(f"Generating thesis for {stock}")

        self.tickr = TickrAgent(
            stocks=[stock],
            max_loops=1,
            workers=10,
            retry_attempts=1,
            context_length=16000,
        )

        try:
            market_data = self.tickr.run(
                f"{task} Analyze current market conditions and key metrics for {stock}"
            )

            prompt = DIRECTOR_THESIS_PROMPT.format(
                task=task, stock=stock, market_data=market_data
            )
            thesis = self.director_agent.run(prompt)
            return thesis, market_data

        except Exception as e:
            logger.error(
                f"Error generating thesis for {stock}: {str(e)}"
            )
            raise

    def make_decision(self, task: str, thesis: str, *args, **kwargs):
        return self.director_agent.run(
            DIRECTOR_DECISION_PROMPT.format(thesis=thesis, task=task)
        )

    def generate_thesis_crypto(
        self,
        task: str = None,
        crypto: str = None,
    ):
        logger.info(f"Generating thesis for {crypto}")
        try:
            market_data = self.crypto_agent.run(
                crypto,
                f"{task} Analyze current market conditions and key metrics for {crypto}",
            )

            prompt = DIRECTOR_THESIS_PROMPT.format(
                task=task, stock=crypto, market_data=market_data
            )
            thesis = self.director_agent.run(prompt)
            return thesis

        except Exception as e:
            logger.error(
                f"Error generating thesis for {crypto}: {str(e)}"
            )
            raise


class QuantAnalyst:
    """
    Quantitative Analysis Agent responsible for technical and statistical analysis.

    Attributes:
        quant_agent (Agent): Swarms agent for analysis
        output_dir (Path): Directory for storing outputs
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Initializing Quant Analyst")
        self.quant_agent = Agent(
            agent_name="Quant-Analyst",
            system_prompt=QUANT_PROMPT,
            model_name="groq/deepseek-r1-distill-llama-70b",
            output_type="str",
            max_loops=1,
            verbose=True,
            context_length=16000,
        )

    def analyze(self, stock: str, thesis: str) -> str:
        """
        Perform quantitative analysis for a stock.

        Args:
            stock (str): Stock ticker symbol
            thesis (TradingThesis): Trading thesis

        Returns:
            QuantAnalysis: Quantitative analysis results
        """
        logger.info(f"Performing quant analysis for {stock}")
        try:
            prompt = QUANT_ANALYSIS_PROMPT.format(
                stock=stock, thesis=thesis
            )
            analysis = self.quant_agent.run(prompt)
            return analysis

        except Exception as e:
            logger.error(
                f"Error in quant analysis for {stock}: {str(e)}"
            )
            raise


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

                self.conversation.add_message(
                    role=self.director.agent_name,
                    content=f"Stock: {stock}\nMarket Data: {market_data}\nThesis: {thesis}",
                )

                # Perform analysis
                analysis = self.quant.analyze(
                    stock + market_data, thesis
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
                    stock + market_data, thesis, analysis
                )

                self.conversation.add(
                    role=self.risk.agent_name, content=risk_assessment
                )

                # # Generate order if approved
                order = self.execution.generate_order(
                    stock, thesis, risk_assessment
                )

                self.conversation.add(
                    role=self.execution.agent_name, content=order
                )

                order = str(order)

                # Final decision
                decision = self.director.make_decision(
                    order + market_data + risk_assessment, thesis
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
