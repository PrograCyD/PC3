package utils

import (
	"fmt"
	"log"
	"os"
	"time"
)

type Logger struct {
	l  *log.Logger
	ts bool
}

func NewLogger(withTimestamp bool) *Logger {
	prefix := ""
	flags := 0
	if withTimestamp {
		flags = log.Lmsgprefix
	}
	return &Logger{
		l:  log.New(os.Stdout, prefix, flags),
		ts: withTimestamp,
	}
}

func (lg *Logger) Info(format string, args ...any) {
	if lg.ts {
		lg.l.Printf("[%s] INFO  %s", time.Now().Format(time.RFC3339), fmt.Sprintf(format, args...))
	} else {
		lg.l.Printf("INFO  %s", fmt.Sprintf(format, args...))
	}
}

func (lg *Logger) Warn(format string, args ...any) {
	if lg.ts {
		lg.l.Printf("[%s] WARN  %s", time.Now().Format(time.RFC3339), fmt.Sprintf(format, args...))
	} else {
		lg.l.Printf("WARN  %s", fmt.Sprintf(format, args...))
	}
}

func (lg *Logger) Error(format string, args ...any) {
	if lg.ts {
		lg.l.Printf("[%s] ERROR %s", time.Now().Format(time.RFC3339), fmt.Sprintf(format, args...))
	} else {
		lg.l.Printf("ERROR %s", fmt.Sprintf(format, args...))
	}
}
