FROM golang:1.26-alpine AS builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o agent-api .

FROM alpine:latest
RUN apk --no-cache add ca-certificates wget
WORKDIR /app
COPY --from=builder /build/agent-api .
COPY --from=builder /build/static ./static

EXPOSE 8080
CMD ["./agent-api"]
