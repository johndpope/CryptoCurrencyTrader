from API_settings import *
from API_settings import API_key, API_secret
from autobahn.asyncio.wamp import ApplicationSession
from autobahn_autoreconnect_custom import ApplicationRunner
from poloniex_API import poloniex
from trollius import coroutine
import trollius as asyncio
import sys
from multiprocessing import Process, Pipe
import logging
logging.basicConfig()


class PoloniexComponent(ApplicationSession):
    def onConnect(self):
        self.join(self.config.realm)

    @coroutine
    def onJoin(self, details):
        def on_ticker(child_conn_local, *args):
            child_conn_local.send(tuple(arg for arg in args))

        child_conn = self.config.extra
        try:
            self.subscribe(lambda *args: on_ticker(child_conn, *args), 'ticker')
        except Exception as e:
            print("Could not subscribe to topic:", e)


def subscribe_to_ticker(child_conn):

    polo = poloniex(API_key, API_secret)

    runner = ApplicationRunner(url=u"wss://api.poloniex.com:443", realm=u"realm1", extra=child_conn)
    runner.run(PoloniexComponent)