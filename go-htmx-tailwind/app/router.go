package app

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
)

type route struct {
	method  string
	regex   *regexp.Regexp
	handler http.Handler
}

type routeFieldCtxKey struct{}

type RouteHandler struct {
	routes []route
}

func (rh *RouteHandler) GetFn(regex string, handler func(http.ResponseWriter, *http.Request)) {
	rh.Get(regex, http.HandlerFunc(handler))
}

func (rh *RouteHandler) Get(regex string, handler http.Handler) {
	rh.registerHandler(http.MethodGet, regex, handler)
}

func (rh *RouteHandler) PostFn(regex string, handler func(http.ResponseWriter, *http.Request)) {
	rh.Post(regex, http.HandlerFunc(handler))
}

func (rh *RouteHandler) Post(regex string, handler http.Handler) {
	rh.registerHandler(http.MethodPost, regex, handler)
}

func (rh *RouteHandler) registerHandler(method string, regex string, handler http.Handler) {
	rh.routes = append(rh.routes, route{method: method, regex: regexp.MustCompile(regex), handler: handler})
}

func (rh *RouteHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// allowedMethods is a slice of allowed HTTP methods for the current route.
	// It is returned if the requested methos is not allowed.
	var allowedMethods []string

	for _, route := range rh.routes {
		matches := route.regex.FindStringSubmatch(r.URL.Path)
		if len(matches) > 0 {
			if r.Method != route.method {
				allowedMethods = append(allowedMethods, route.method)
				continue
			}
			ctx := context.WithValue(r.Context(), routeFieldCtxKey{}, matches[1:])
			route.handler.ServeHTTP(w, r.WithContext(ctx))
			return
		}
	}
	if len(allowedMethods) > 0 {
		w.Header().Set("Allow", strings.Join(allowedMethods, ", "))
		http.Error(w, fmt.Sprintf("'%v' method not allowed.", r.Method), http.StatusMethodNotAllowed)
		return
	}
	http.NotFound(w, r)
}

func GetRouteParam(r *http.Request, index int) (string, error) {
	params, ok := r.Context().Value(routeFieldCtxKey{}).([]string)
	if !ok {
		return "", errors.New("failed to get route params")
	}

	if index >= len(params) {
		return "", fmt.Errorf("route param index '%v' is out of range", index)
	}

	return params[index], nil
}
