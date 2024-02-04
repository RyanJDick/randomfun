package main

import (
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ryanjdick/go-htmx-tailwind/app/handlers"
)

func main() {

	r := gin.Default()
	r.LoadHTMLGlob("templates/*")

	logger := slog.Default()

	hh := &handlers.HelloHandler{Logger: logger}

	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "pong",
		})
	})
	r.GET("/hello/:name", hh.GetHello)
	r.GET("/time", hh.GetTime)

	r.StaticFile("/", "static/index.html")
	r.Static("/static", "static/")

	r.Run() // listen and serve on 0.0.0.0:8080 (for windows "localhost:8080")
}
