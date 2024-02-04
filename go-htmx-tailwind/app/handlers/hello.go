package handlers

import (
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"
)

type helloData struct {
	Name string
}

type HelloHandler struct {
	Logger *slog.Logger
}

func (hh *HelloHandler) GetHello(c *gin.Context) {
	name := c.Param("name")
	c.HTML(http.StatusOK, "hello.tmpl", helloData{Name: name})
}
