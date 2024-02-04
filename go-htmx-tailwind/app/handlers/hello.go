package handlers

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type helloData struct {
	Name string
	Time string
}

type HelloHandler struct {
	Logger *slog.Logger
}

func getTime() string {
	return time.Now().Format(time.RFC1123)
}

func (hh *HelloHandler) GetHello(c *gin.Context) {
	name := c.Param("name")
	c.HTML(http.StatusOK, "hello.html", helloData{Name: name, Time: getTime()})
}

func (hh *HelloHandler) GetTime(c *gin.Context) {
	c.HTML(http.StatusOK, "hello.time", gin.H{"Time": getTime()})
}
