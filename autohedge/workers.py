"""
AutoHedge workers: Pydantic output models and agent classes for thesis
generation, risk assessment, execution, and quantitative analysis.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from swarms import Agent

from autohedge.prompts import (
    DIRECTOR_DECISION_PROMPT,
    DIRECTOR_PROMPT,
    DIRECTOR_THESIS_PROMPT,
    EXECUTION_ORDER_PROMPT,
    EXECUTION_PROMPT,
    QUANT_ANALYSIS_PROMPT,
    QUANT_PROMPT,
    RISK_ASSESSMENT_PROMPT,
    RISK_PROMPT,
    SENTIMENT_PROMPT,
)
from autohedge.tools.yahoo_api import get_all_stock_data

sentiment_agent = Agent(
    agent_name="Sentiment-Agent",
    system_prompt=SENTIMENT_PROMPT,
    model_name="gpt-4o-mini",
    output_type="str",
    max_loops=1,
    verbose=True,
    context_length=16000,
)


def _agent_context(task: Optional[str] = None) -> str:
    """Build context string with current time and task to prepend to agent prompts."""
    now = datetime.now().isoformat()
    task_str = task if task else "(none)"
    return f"Current time: {now}\nTask: {task_str}\n\n"


class AutoHedgeOutput(BaseModel):
    """
    Per-stock output from the AutoHedge pipeline for a single ticker.

    Attributes
    ----------
    id : str
        Unique run identifier (hex UUID).
    thesis : str, optional
        Trading thesis for the stock.
    risk_assessment : str, optional
        Risk assessment text from the risk manager.
    order : str, optional
        Generated order / execution plan.
    decision : str, optional
        Director decision (e.g. hold, buy, sell).
    timestamp : str
        ISO timestamp when this output was produced.
    current_stock : str
        Ticker symbol this output refers to.
    """

    id: str = uuid.uuid4().hex
    thesis: Optional[str] = None
    risk_assessment: Optional[str] = None
    order: Optional[str] = None
    decision: str = None
    timestamp: str = datetime.now().isoformat()
    current_stock: str


class AutoHedgeOutputMain(BaseModel):
    """
    Top-level output from a full AutoHedge run (multiple stocks / task).

    Attributes
    ----------
    name : str, optional
        Run or strategy name.
    description : str, optional
        Human-readable description of the run.
    id : str
        Unique run identifier (hex UUID).
    stocks : list, optional
        List of ticker symbols processed.
    task : str, optional
        User task or instruction for the run.
    timestamp : str
        ISO timestamp when the run completed.
    logs : list of AutoHedgeOutput, optional
        Per-stock results in order of processing.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    id: str = uuid.uuid4().hex
    stocks: Optional[list] = None
    task: Optional[str] = None
    timestamp: str = datetime.now().isoformat()
    logs: List[AutoHedgeOutput] = None


class RiskManager:
    """
    Agent that assesses risk for a stock given a thesis and quant analysis.

    Uses a dedicated Swarms agent (RISK_PROMPT) to produce a text risk
    assessment from the trading thesis and quantitative analysis.

    Attributes
    ----------
    risk_agent : Agent
        Swarms agent used for risk assessment.
    """

    def __init__(self):
        self.risk_agent = Agent(
            agent_name="Risk-Manager",
            system_prompt=RISK_PROMPT,
            model_name="gpt-4.1",
            output_type="str",
            max_loops=1,
            verbose=True,
            context_length=16000,
        )

    def assess_risk(
        self,
        stock: str,
        thesis: str,
        quant_analysis: str,
        task: Optional[str] = None,
    ) -> str:
        """
        Produce a risk assessment for a stock given thesis and quant analysis.

        Parameters
        ----------
        stock : str
            Ticker symbol.
        thesis : str
            Trading thesis text.
        quant_analysis : str
            Quantitative analysis text.
        task : str, optional
            User task or instruction (included in agent context with current time).

        Returns
        -------
        str
            Risk assessment text from the risk agent.
        """
        prompt = _agent_context(task) + RISK_ASSESSMENT_PROMPT.format(
            stock=stock, thesis=thesis, quant_analysis=quant_analysis
        )
        assessment = self.risk_agent.run(prompt)

        return assessment


class ExecutionAgent:
    """
    Agent that generates execution orders from thesis and risk assessment.

    Uses a Swarms agent (EXECUTION_PROMPT) to turn a thesis and risk
    assessment into a concrete order or execution plan.

    Attributes
    ----------
    execution_agent : Agent
        Swarms agent used for order generation.
    """

    def __init__(self):
        self.execution_agent = Agent(
            agent_name="Execution-Agent",
            system_prompt=EXECUTION_PROMPT,
            model_name="gpt-4.1",
            output_type="str",
            max_loops=1,
            verbose=True,
            context_length=16000,
        )

    def generate_order(
        self,
        stock: str,
        thesis: Dict,
        risk_assessment: Dict,
        task: Optional[str] = None,
    ) -> str:
        """
        Generate an execution order for a stock from thesis and risk data.

        Parameters
        ----------
        stock : str
            Ticker symbol.
        thesis : dict
            Trading thesis (or serialized thesis data).
        risk_assessment : dict
            Risk assessment (or serialized risk data).
        task : str, optional
            User task or instruction (included in agent context with current time).

        Returns
        -------
        str
            Generated order text from the execution agent.
        """
        prompt = _agent_context(task) + EXECUTION_ORDER_PROMPT.format(
            stock=stock,
            thesis=thesis,
            risk_assessment=risk_assessment,
        )
        order = self.execution_agent.run(prompt)
        return order


class TradingDirector:
    """
    Coordinates strategy and generates trading theses using market data.

    Uses a Swarms director agent and Yahoo Finance data to fetch market data,
    then produces a trading thesis and can make follow-up decisions.

    Attributes
    ----------
    director_agent : Agent
        Swarms agent for thesis generation and decisions.
    """

    def __init__(
        self,
        stocks: List[str],
        output_dir: str = "outputs",
        cryptos: List[str] = None,
    ):
        """
        Parameters
        ----------
        stocks : list of str
            Ticker symbols the director may analyze.
        output_dir : str, optional
            Directory for outputs (default "outputs").
        cryptos : list of str, optional
            Crypto symbols for crypto thesis (currently unused).
        """
        logger.info("Initializing Trading Director")
        self.director_agent = Agent(
            agent_name="Trading-Director",
            system_prompt=DIRECTOR_PROMPT,
            model_name="gpt-4.1",
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

        try:
            market_data = get_all_stock_data(
                stock, include_history=True
            )

            prompt = _agent_context(
                task
            ) + DIRECTOR_THESIS_PROMPT.format(
                task=task, stock=stock, market_data=market_data
            )
            thesis = self.director_agent.run(prompt)
            return thesis, market_data

        except Exception as e:
            logger.error(
                f"Error generating thesis for {stock}: {str(e)}"
            )
            raise

    def make_decision(
        self,
        task: str,
        thesis: str,
        user_task: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Run the director agent to make a decision given a task and thesis.

        Parameters
        ----------
        task : str
            Order/context to evaluate (e.g. order + market_data + risk_assessment).
        thesis : str
            Trading thesis text.
        user_task : str, optional
            User task or instruction (included in agent context with current time).
        *args, **kwargs
            Passed through to the agent (e.g. for future options).

        Returns
        -------
        str
            Director decision output.
        """
        prompt = _agent_context(
            user_task
        ) + DIRECTOR_DECISION_PROMPT.format(thesis=thesis, task=task)
        return self.director_agent.run(prompt)

    def generate_thesis_crypto(
        self,
        task: str = None,
        crypto: str = None,
    ):
        """
        Generate a trading thesis for a crypto asset using the crypto agent.

        Parameters
        ----------
        task : str, optional
            Analysis task or instruction.
        crypto : str, optional
            Crypto symbol (e.g. BTC, ETH).

        Returns
        -------
        str
            Generated thesis text.

        Raises
        ------
        Exception
            If crypto_agent is not set or the run fails.
        """
        logger.info(f"Generating thesis for {crypto}")
        try:
            market_data = self.crypto_agent.run(
                crypto,
                f"{task} Analyze current market conditions and key metrics for {crypto}",
            )

            prompt = _agent_context(
                task
            ) + DIRECTOR_THESIS_PROMPT.format(
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
    Agent that performs quantitative (technical and statistical) analysis.

    Uses a Swarms agent (QUANT_PROMPT) to analyze a stock in the context
    of a trading thesis and produce structured quant analysis.

    Attributes
    ----------
    quant_agent : Agent
        Swarms agent used for analysis.
    output_dir : Path
        Directory for saving outputs (created on init).
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Initializing Quant Analyst")
        self.quant_agent = Agent(
            agent_name="Quant-Analyst",
            system_prompt=QUANT_PROMPT,
            model_name="gpt-4.1",
            output_type="str",
            max_loops=1,
            verbose=True,
            context_length=16000,
        )

    def analyze(
        self, stock: str, thesis: str, task: Optional[str] = None
    ) -> str:
        """
        Perform quantitative analysis for a stock given a trading thesis.

        Parameters
        ----------
        stock : str
            Stock ticker symbol.
        thesis : str
            Trading thesis text.
        task : str, optional
            User task or instruction (included in agent context with current time).

        Returns
        -------
        str
            Quantitative analysis text from the quant agent.
        """
        logger.info(f"Performing quant analysis for {stock}")
        try:
            prompt = _agent_context(
                task
            ) + QUANT_ANALYSIS_PROMPT.format(
                stock=stock, thesis=thesis
            )
            analysis = self.quant_agent.run(prompt)
            return analysis

        except Exception as e:
            logger.error(
                f"Error in quant analysis for {stock}: {str(e)}"
            )
            raise


# -----------------------------------------------------------------------------
# Initialized workers and their agents (for discovery / iteration)
# -----------------------------------------------------------------------------
risk_manager = RiskManager()
execution_agent_instance = ExecutionAgent()
trading_director = TradingDirector(stocks=[])
quant_analyst = QuantAnalyst()

ALL_AGENTS = [
    sentiment_agent,  # Sentiment-Agent
    risk_manager.risk_agent,  # Risk-Manager
    execution_agent_instance.execution_agent,  # Execution-Agent
    trading_director.director_agent,  # Trading-Director
    quant_analyst.quant_agent,  # Quant-Analyst
]
